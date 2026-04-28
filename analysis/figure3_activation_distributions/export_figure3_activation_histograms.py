from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/"
    / "20260114184203_default_acc=53.50/best_checkpoint.pth"
)
DEFAULT_DATA_ROOT = ROOT / "data/vtab-1k"
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "csv/figure3_activation_histograms.csv"
DEFAULT_SUMMARY_CSV = Path(__file__).resolve().parent / "csv/figure3_activation_summary.csv"
CLASS_COUNTS = {
    "cifar": 100,
    "caltech101": 102,
    "dtd": 47,
    "oxford_flowers102": 102,
    "oxford_iiit_pet": 37,
    "svhn": 10,
    "sun397": 397,
    "patch_camelyon": 2,
    "eurosat": 10,
    "resisc45": 45,
    "diabetic_retinopathy": 5,
    "clevr_count": 8,
    "clevr_dist": 6,
    "dmlab": 6,
    "kitti": 4,
    "dsprites_loc": 16,
    "dsprites_ori": 16,
    "smallnorb_azi": 18,
    "smallnorb_ele": 9,
}


class CupyFallback(types.ModuleType):
    """Small CPU-backed cupy fallback for this analysis script.

    The QPEFT quantizer imports cupy unconditionally, but the final checkpoint
    already contains quantizer scales. This fallback keeps imports and any rare
    percentile initialization path working without changing project code.
    """

    def __init__(self) -> None:
        super().__init__("cupy")

    @staticmethod
    def asarray(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def percentile(a, q, axis=None, keepdims=False, overwrite_input=False, method="linear"):
        return np.percentile(
            a,
            q,
            axis=axis,
            keepdims=keepdims,
            overwrite_input=overwrite_input,
            method=method,
        )


def ensure_cupy_importable() -> None:
    try:
        import cupy  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["cupy"] = CupyFallback()


class VTABImageListDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: Path, dataset: str, split: str, max_images: int, transform) -> None:
        self.root = data_root / dataset
        self.transform = transform
        list_path = self.root / f"{split}.txt"
        if not list_path.exists():
            raise FileNotFoundError(f"VTAB split list not found: {list_path}")

        self.samples: list[tuple[Path, int]] = []
        with list_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(self.samples) >= max_images:
                    break
                image_name, label = line.strip().split()[:2]
                image_path = self.root / image_name
                if not image_path.exists():
                    raise FileNotFoundError(f"Image listed in {list_path} does not exist: {image_path}")
                self.samples.append((image_path, int(label)))
        if not self.samples:
            raise ValueError(f"No samples loaded from {list_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


class ActivationCollector:
    def __init__(self, max_values: int, per_hook_values: int, seed: int) -> None:
        self.max_values = max_values
        self.per_hook_values = per_hook_values
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.values: dict[str, list[torch.Tensor]] = {
            "ordinary": [],
            "post-GELU": [],
            "post-softmax": [],
        }

    def hook_for(self, tensor_type: str):
        def hook(_module, inputs):
            tensor = inputs[0].detach().float().flatten()
            if tensor.numel() > self.per_hook_values:
                indices = torch.randint(
                    high=tensor.numel(),
                    size=(self.per_hook_values,),
                    generator=self.generator,
                )
                if tensor.device.type != "cpu":
                    indices = indices.to(tensor.device)
                tensor = tensor[indices]
            self.values[tensor_type].append(tensor.cpu())

        return hook

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {}
        for tensor_type, chunks in self.values.items():
            if not chunks:
                raise ValueError(f"No activations captured for tensor type: {tensor_type}")
            values = torch.cat(chunks).numpy()
            if values.size > self.max_values:
                rng = np.random.default_rng(0)
                values = values[rng.choice(values.size, size=self.max_values, replace=False)]
            arrays[tensor_type] = values.astype(np.float64, copy=False)
        return arrays


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )


def build_qpeft_model(args: argparse.Namespace) -> torch.nn.Module:
    ensure_cupy_importable()

    import models
    from configs import MODELS, load_configs
    from qpeft_quant.wrap_net_qpeft import apply_qpeft

    config_args = SimpleNamespace(train_cfgs=args.train_cfgs, quant_cfgs=args.quant_cfgs, lora_cfgs=args.lora_cfgs)
    cfgs = load_configs(config_args)
    for key, value in vars(config_args).items():
        setattr(cfgs, key, value)
    cfgs.model = args.model
    cfgs.dataset = args.dataset
    cfgs.device = args.device
    cfgs.inception = True

    if args.dataset not in CLASS_COUNTS:
        raise ValueError(f"Unknown class count for dataset {args.dataset!r}")

    model = models.__dict__[MODELS[cfgs.model].full_name](drop_path_rate=cfgs.drop_path)
    model.reset_classifier(CLASS_COUNTS[args.dataset])
    apply_qpeft(model, cfgs)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a 'model' state dict: {args.checkpoint}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        available = [module_name for module_name in modules if module_name.endswith("act_quantizer")]
        raise KeyError(f"Module {name!r} not found. Example available activation modules: {available[:12]}")
    return modules[name]


def capture_activations(args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = torch.device(args.device)
    model = build_qpeft_model(args).to(device)

    collector = ActivationCollector(
        max_values=args.max_values,
        per_hook_values=args.per_hook_values,
        seed=args.seed,
    )
    block = args.block_index
    hook_specs = {
        "ordinary": f"blocks.{block}.attn.qkv.act_quantizer",
        "post-GELU": f"blocks.{block}.mlp.fc2.act_quantizer",
        "post-softmax": f"blocks.{block}.attn.attn_quantizer",
    }
    handles = [
        get_module(model, module_name).register_forward_pre_hook(collector.hook_for(tensor_type))
        for tensor_type, module_name in hook_specs.items()
    ]

    dataset = VTABImageListDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        max_images=args.max_images,
        transform=build_transform(),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for images, _labels in loader:
            model(images.to(device))

    for handle in handles:
        handle.remove()

    arrays = collector.arrays()
    metadata = {
        "checkpoint": str(args.checkpoint.relative_to(ROOT) if args.checkpoint.is_relative_to(ROOT) else args.checkpoint),
        "dataset": args.dataset,
        "split": args.split,
        "max_images": args.max_images,
        "batch_size": args.batch_size,
        "block_index": block,
        "hook_modules": hook_specs,
    }
    (args.output_csv.parent / "figure3_activation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return arrays


def histogram_range(tensor_type: str, values: np.ndarray) -> tuple[float, float]:
    if tensor_type == "post-softmax":
        return 0.0, float(min(1.0, max(np.quantile(values, 0.999), 1e-6)))

    low, high = np.quantile(values, [0.001, 0.999])
    if low == high:
        low, high = float(values.min()), float(values.max())
    if low == high:
        high = low + 1e-6
    return float(low), float(high)


def write_histograms(arrays: dict[str, np.ndarray], output_csv: Path, bins: int) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tensor_type in ["ordinary", "post-GELU", "post-softmax"]:
        values = arrays[tensor_type]
        low, high = histogram_range(tensor_type, values)
        counts, edges = np.histogram(values, bins=bins, range=(low, high), density=False)
        width = np.diff(edges)
        in_range_count = counts.sum()
        density = counts / max(in_range_count, 1) / width
        for idx, count in enumerate(counts):
            rows.append(
                {
                    "tensor_type": tensor_type,
                    "bin_left": edges[idx],
                    "bin_right": edges[idx + 1],
                    "density": density[idx],
                    "count": int(count),
                    "in_range_count": int(in_range_count),
                    "total_sampled_count": int(values.size),
                    "plot_range_low": low,
                    "plot_range_high": high,
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(arrays: dict[str, np.ndarray], output_csv: Path) -> None:
    rows = []
    for tensor_type in ["ordinary", "post-GELU", "post-softmax"]:
        values = arrays[tensor_type]
        rows.append(
            {
                "tensor_type": tensor_type,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
            }
        )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real activation histograms for Figure 3 from the QPEFT ViT checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--split", default="test", choices=["train800", "train800val200", "val200", "test"])
    parser.add_argument("--model", default="vitb16")
    parser.add_argument("--train-cfgs", default="vtab")
    parser.add_argument("--quant-cfgs", default="3bit")
    parser.add_argument("--lora-cfgs", default="lora_2_qkv-fc1-proj-fc2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block-index", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-values", type=int, default=300_000)
    parser.add_argument("--per-hook-values", type=int, default=80_000)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = capture_activations(args)
    write_histograms(arrays, args.output_csv, args.bins)
    write_summary(arrays, args.summary_csv)
    print(f"Wrote Figure 3 histogram CSV: {args.output_csv}")
    print(f"Wrote Figure 3 summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
