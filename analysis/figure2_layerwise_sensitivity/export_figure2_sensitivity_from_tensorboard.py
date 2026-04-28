from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_FILE = (
    ROOT
    / "outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/"
    / "20260114184203_default_acc=53.50/tensorboard/"
    / "events.out.tfevents.1768387332.master-192-168-8-48.1680.1"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "csv/figure2_layerwise_sensitivity.csv"


@dataclass(frozen=True)
class ModulePattern:
    module_name: str
    pattern: re.Pattern[str]


MODULE_PATTERNS = [
    ModulePattern("qkv", re.compile(r"^blocks\.(\d+)\.attn\.qkv\.weight_quantizer$")),
    ModulePattern("proj", re.compile(r"^blocks\.(\d+)\.attn\.proj\.weight_quantizer$")),
    ModulePattern("fc1", re.compile(r"^blocks\.(\d+)\.mlp\.fc1\.weight_quantizer$")),
    ModulePattern("fc2", re.compile(r"^blocks\.(\d+)\.mlp\.fc2\.weight_quantizer$")),
    ModulePattern("post-GELU", re.compile(r"^blocks\.(\d+)\.mlp\.fc2\.act_quantizer$")),
    ModulePattern("post-softmax", re.compile(r"^blocks\.(\d+)\.attn\.attn_quantizer$")),
]


def load_event_scalars(event_file: Path) -> EventAccumulator:
    if not event_file.exists():
        raise FileNotFoundError(f"TensorBoard event file not found: {event_file}")

    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    if "scalars" not in accumulator.Tags() or not accumulator.Tags()["scalars"]:
        raise ValueError(f"No scalar tags found in TensorBoard event file: {event_file}")
    return accumulator


def match_module(tag: str) -> tuple[int, str] | None:
    for module_pattern in MODULE_PATTERNS:
        match = module_pattern.pattern.match(tag)
        if match:
            return int(match.group(1)), module_pattern.module_name
    return None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def export_csv(event_file: Path, output_csv: Path) -> None:
    accumulator = load_event_scalars(event_file)
    rows = []

    for tag in accumulator.Tags()["scalars"]:
        matched = match_module(tag)
        if matched is None:
            continue

        layer_index, module_name = matched
        scalars = accumulator.Scalars(tag)
        if not scalars:
            raise ValueError(f"Scalar tag has no values: {tag}")
        final_scalar = scalars[-1]
        rows.append(
            {
                "layer_index": layer_index,
                "module_name": module_name,
                "sensitivity_value": final_scalar.value,
                "source_metric": "final_adam_exp_avg_sq_mean",
                "source_step": final_scalar.step,
                "source_tag": tag,
                "source_event_file": display_path(event_file),
            }
        )

    expected_modules = {module_pattern.module_name for module_pattern in MODULE_PATTERNS}
    observed_modules = {row["module_name"] for row in rows}
    observed_blocks = {int(row["layer_index"]) for row in rows}
    expected_count = len(expected_modules) * len(observed_blocks)
    if not rows:
        raise ValueError(
            "No Figure 2 sensitivity rows were extracted. "
            "Expected tags such as blocks.0.attn.qkv.weight_quantizer."
        )
    if observed_modules != expected_modules:
        missing = sorted(expected_modules - observed_modules)
        raise ValueError(f"Missing required Figure 2 module types in event file: {missing}")
    if len(rows) != expected_count:
        raise ValueError(
            f"Extracted {len(rows)} rows, but expected {expected_count} "
            f"({len(observed_blocks)} blocks x {len(expected_modules)} modules)."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda item: (int(item["layer_index"]), item["module_name"]))
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Figure 2 layerwise sensitivity CSV from the QPEFT TensorBoard "
            "second-moment scalar event file."
        )
    )
    parser.add_argument("--event-file", type=Path, default=DEFAULT_EVENT_FILE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_csv(args.event_file, args.output_csv)
    print(f"Wrote Figure 2 sensitivity CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
