from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_HISTOGRAM_CSV = Path(__file__).resolve().parent / "csv/figure3_activation_histograms.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_STEM = "figure3_activation_distribution_comparison"
REQUIRED_HISTOGRAM_COLUMNS = {"tensor_type", "bin_left", "bin_right", "density", "count"}
TENSOR_ORDER = ["ordinary", "post-GELU", "post-softmax"]
TENSOR_LABELS = {
    "ordinary": "Ordinary activations",
    "post-GELU": "post-GELU activations",
    "post-softmax": "post-softmax attention",
}
X_LABELS = {
    "ordinary": "Activation value",
    "post-GELU": "Activation value",
    "post-softmax": "Attention probability",
}
COLORS = {
    "ordinary": "#315C8A",
    "post-GELU": "#B45A3C",
    "post-softmax": "#3B7A57",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.3,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
        }
    )


def load_histogram_csv(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Histogram CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Histogram CSV has no header row: {path}")
        missing = sorted(REQUIRED_HISTOGRAM_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Histogram CSV is missing required columns: {missing}. "
                f"Required columns are: {sorted(REQUIRED_HISTOGRAM_COLUMNS)}"
            )
        grouped = defaultdict(lambda: {"left": [], "right": [], "density": [], "count": []})
        for line_number, row in enumerate(reader, start=2):
            tensor_type = row["tensor_type"].strip()
            if tensor_type not in TENSOR_ORDER:
                raise ValueError(f"Unknown tensor_type on line {line_number}: {tensor_type!r}")
            try:
                grouped[tensor_type]["left"].append(float(row["bin_left"]))
                grouped[tensor_type]["right"].append(float(row["bin_right"]))
                grouped[tensor_type]["density"].append(float(row["density"]))
                grouped[tensor_type]["count"].append(float(row["count"]))
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value in histogram CSV line {line_number}") from exc

    missing_types = [tensor_type for tensor_type in TENSOR_ORDER if tensor_type not in grouped]
    if missing_types:
        raise ValueError(f"Histogram CSV is missing tensor types: {missing_types}")

    return {
        tensor_type: {key: np.asarray(values, dtype=float) for key, values in values_by_key.items()}
        for tensor_type, values_by_key in grouped.items()
    }


def load_dump(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Activation dump not found: {path}")
    if path.suffix == ".npy":
        values = np.load(path)
    elif path.suffix == ".npz":
        loaded = np.load(path)
        if len(loaded.files) != 1:
            raise ValueError(f"NPZ dump must contain exactly one array: {path}")
        values = loaded[loaded.files[0]]
    elif path.suffix in {".pt", ".pth"}:
        values = torch.load(path, map_location="cpu")
        if isinstance(values, dict):
            if len(values) != 1:
                raise ValueError(f"Torch dump dict must contain exactly one tensor: {path}")
            values = next(iter(values.values()))
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported dump format for {path}; use .npy, .npz, .pt, or .pth")
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Activation dump contains no finite values: {path}")
    return values


def histogram_from_dumps(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    dump_paths = {
        "ordinary": args.ordinary_dump,
        "post-GELU": args.post_gelu_dump,
        "post-softmax": args.post_softmax_dump,
    }
    if any(path is None for path in dump_paths.values()):
        raise ValueError(
            "Provide --histogram-csv, or provide all three dump paths: "
            "--ordinary-dump, --post-gelu-dump, and --post-softmax-dump."
        )

    grouped = {}
    for tensor_type, path in dump_paths.items():
        values = load_dump(path)
        low, high = np.quantile(values, [0.001, 0.999])
        if tensor_type == "post-softmax":
            low = 0.0
            high = min(1.0, max(float(high), 1e-6))
        if low == high:
            low, high = float(values.min()), float(values.max())
        if low == high:
            high = low + 1e-6
        counts, edges = np.histogram(values, bins=args.bins, range=(low, high), density=False)
        widths = np.diff(edges)
        density = counts / max(counts.sum(), 1) / widths
        grouped[tensor_type] = {
            "left": edges[:-1],
            "right": edges[1:],
            "density": density,
            "count": counts.astype(float),
        }
    return grouped


def demo_histograms(bins: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    synthetic = {
        "ordinary": rng.normal(0.0, 0.85, 200_000),
        "post-GELU": rng.gamma(shape=1.4, scale=0.55, size=200_000) - 0.17,
        "post-softmax": rng.beta(a=0.32, b=35.0, size=200_000),
    }
    grouped = {}
    for tensor_type, values in synthetic.items():
        low, high = np.quantile(values, [0.001, 0.999])
        if tensor_type == "post-softmax":
            low = 0.0
        counts, edges = np.histogram(values, bins=bins, range=(low, high), density=False)
        density = counts / max(counts.sum(), 1) / np.diff(edges)
        grouped[tensor_type] = {
            "left": edges[:-1],
            "right": edges[1:],
            "density": density,
            "count": counts.astype(float),
        }
    return grouped


def save_multi_format(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=600)


def plot_histograms(grouped: dict[str, dict[str, np.ndarray]], output_dir: Path, stem: str, demo: bool) -> None:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35), constrained_layout=True)

    for ax, tensor_type in zip(axes, TENSOR_ORDER):
        values = grouped[tensor_type]
        centers = 0.5 * (values["left"] + values["right"])
        density = values["density"]
        color = COLORS[tensor_type]

        ax.fill_between(centers, density, color=color, alpha=0.23, linewidth=0)
        ax.plot(centers, density, color=color, linewidth=1.65)
        ax.set_title(TENSOR_LABELS[tensor_type])
        ax.set_xlabel(X_LABELS[tensor_type])
        ax.grid(True, axis="y", color="#D8DCE2", linewidth=0.55, alpha=0.72)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")
        if tensor_type == "post-softmax":
            ax.set_xlim(left=0.0)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        else:
            ax.axvline(0.0, color="#222222", linewidth=0.75, alpha=0.45)
        if np.nanmax(density) / max(np.nanmedian(density[density > 0]), 1e-12) > 1_000:
            ax.set_yscale("log")
            ax.set_ylabel("Density (log)")
        else:
            ax.set_ylabel("Density")

    if demo:
        fig.text(0.995, 0.01, "DEMO DATA - DO NOT USE IN REPORT", ha="right", va="bottom", fontsize=6.8, color="#9A3412")

    save_multi_format(fig, output_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Figure 3 activation distributions from real histogram CSV or activation dump files. "
            "Synthetic distributions are generated only when --demo is explicitly set."
        )
    )
    parser.add_argument("--histogram-csv", type=Path, default=DEFAULT_HISTOGRAM_CSV)
    parser.add_argument("--ordinary-dump", type=Path)
    parser.add_argument("--post-gelu-dump", type=Path)
    parser.add_argument("--post-softmax-dump", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        grouped = demo_histograms(args.bins)
    elif args.histogram_csv.exists():
        grouped = load_histogram_csv(args.histogram_csv)
    else:
        grouped = histogram_from_dumps(args)
    plot_histograms(grouped, args.output_dir, args.stem, args.demo)
    print(f"Wrote Figure 3 distribution files to: {args.output_dir}")


if __name__ == "__main__":
    main()
