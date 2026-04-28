from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterMathtext


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(__file__).resolve().parent / "csv/figure2_layerwise_sensitivity.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_STEM = "figure2_layerwise_sensitivity_heatmap"
REQUIRED_COLUMNS = {"layer_index", "module_name", "sensitivity_value"}
MODULE_ORDER = ["qkv", "proj", "fc1", "fc2", "post-GELU", "post-softmax"]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
        }
    )


def read_sensitivity_csv(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Sensitivity CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Sensitivity CSV has no header row: {csv_path}")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Sensitivity CSV is missing required columns: {missing}. "
                f"Required columns are: {sorted(REQUIRED_COLUMNS)}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                layer_index = int(row["layer_index"])
            except ValueError as exc:
                raise ValueError(f"Invalid layer_index on CSV line {line_number}: {row['layer_index']!r}") from exc
            module_name = row["module_name"].strip()
            if not module_name:
                raise ValueError(f"Empty module_name on CSV line {line_number}")
            try:
                value = float(row["sensitivity_value"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid sensitivity_value on CSV line {line_number}: {row['sensitivity_value']!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite sensitivity_value on CSV line {line_number}: {value!r}")
            rows.append({"layer_index": layer_index, "module_name": module_name, "sensitivity_value": value})

    if not rows:
        raise ValueError(f"Sensitivity CSV contains no data rows: {csv_path}")
    return rows


def build_matrix(rows: list[dict[str, object]]) -> tuple[np.ndarray, list[int], list[str]]:
    observed_modules = {str(row["module_name"]) for row in rows}
    module_order = [module for module in MODULE_ORDER if module in observed_modules]
    module_order.extend(sorted(observed_modules - set(module_order)))
    blocks = sorted({int(row["layer_index"]) for row in rows})

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["layer_index"]), str(row["module_name"]))
        grouped[key].append(float(row["sensitivity_value"]))

    matrix = np.full((len(blocks), len(module_order)), np.nan, dtype=float)
    for block_idx, block in enumerate(blocks):
        for module_idx, module in enumerate(module_order):
            values = grouped.get((block, module), [])
            if values:
                matrix[block_idx, module_idx] = float(np.mean(values))

    if np.isnan(matrix).any():
        missing = [
            (block, module_order[module_idx])
            for block_idx, block in enumerate(blocks)
            for module_idx in range(len(module_order))
            if np.isnan(matrix[block_idx, module_idx])
        ]
        raise ValueError(f"Sensitivity CSV is missing block/module combinations: {missing[:12]}")

    return matrix, blocks, module_order


def save_multi_format(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=600)


def plot_heatmap(
    matrix: np.ndarray,
    blocks: list[int],
    modules: list[str],
    output_dir: Path,
    stem: str,
    scale: str,
) -> None:
    set_style()

    if scale == "log":
        if np.any(matrix <= 0):
            raise ValueError("Log-scale heatmap requires strictly positive sensitivity values.")
        norm = LogNorm(vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
        colorbar_kwargs = {"format": LogFormatterMathtext()}
    elif scale == "linear":
        norm = Normalize(vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
        colorbar_kwargs = {}
    else:
        raise ValueError(f"Unsupported scale: {scale!r}")

    fig_width = max(4.9, 0.62 * len(modules) + 1.9)
    fig_height = max(3.0, 0.22 * len(blocks) + 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="cividis", norm=norm)
    ax.set_xlabel("Module type")
    ax.set_ylabel("Transformer block index")
    ax.set_xticks(np.arange(len(modules)))
    ax.set_xticklabels(modules)
    ax.set_yticks(np.arange(len(blocks)))
    ax.set_yticklabels([str(block) for block in blocks])

    ax.set_xticks(np.arange(-0.5, len(modules), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(blocks), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65, alpha=0.88)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035, **colorbar_kwargs)
    colorbar.set_label("Sensitivity")
    colorbar.outline.set_linewidth(0.6)

    save_multi_format(fig, output_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 2 layerwise sensitivity heatmap from a CSV file using matplotlib."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--scale", choices=["log", "linear"], default="log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_sensitivity_csv(args.input_csv)
    matrix, blocks, modules = build_matrix(rows)
    plot_heatmap(matrix, blocks, modules, args.output_dir, args.stem, args.scale)
    print(f"Wrote Figure 2 heatmap files to: {args.output_dir}")


if __name__ == "__main__":
    main()
