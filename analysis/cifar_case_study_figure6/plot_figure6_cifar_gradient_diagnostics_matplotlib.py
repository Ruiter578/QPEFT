from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "analysis/tensorboard_cifar_3bit_qpeft/csv/gradients_selected_long.csv"
DEFAULT_EVENT_ROOT = (
    ROOT
    / "outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/"
    / "20260114184203_default_acc=53.50/tensorboard"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figures"
STEPS_PER_EPOCH = 32.0

BLOCKS = [0, 5, 11]
SERIES = {
    "attn_quantizer.log_base_alpha": {
        "tag": "Grad/attn_quantizer.log_base_alpha",
        "event_dir": "Grad_attn_quantizer.log_base_alpha_block_{block}",
        "title": "A. Quantizer log-base alpha",
        "gradient_name": "Grad_attn_quantizer.log_base_alpha",
    },
    "qkv.lora_B": {
        "tag": "Grad/qkv.lora_B",
        "event_dir": "Grad_qkv.lora_B_block_{block}",
        "title": "B. qkv LoRA-B",
        "gradient_name": "Grad_qkv.lora_B",
    },
    "mlp.fc2.lora_B": {
        "tag": "Grad/mlp.fc2.lora_B",
        "event_dir": "Grad_mlp.fc2.lora_B_block_{block}",
        "title": "C. mlp.fc2 LoRA-B",
        "gradient_name": "Grad_mlp.fc2.lora_B",
    },
}

COLORS = {
    0: "#365f91",
    5: "#c76d1d",
    11: "#3f7f4a",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 6 gradient diagnostics from real TensorBoard scalars or exported CSV data."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Exported long-form gradient scalar CSV.")
    parser.add_argument(
        "--event-root",
        type=Path,
        default=DEFAULT_EVENT_ROOT,
        help="TensorBoard directory containing Grad_*_block_* event subdirectories.",
    )
    parser.add_argument(
        "--prefer-events",
        action="store_true",
        help="Read TensorBoard event files even if the exported CSV exists.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for final figures.")
    parser.add_argument("--stem", default="figure6_cifar_gradient_diagnostics", help="Output filename stem.")
    parser.add_argument("--smooth-span", type=positive_int, default=81, help="EMA span for the trend traces.")
    return parser.parse_args()


def read_tensorboard_events(event_root: Path) -> pd.DataFrame:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("Reading event files requires tensorboard. Install it or provide --csv.") from exc

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for parameter, meta in SERIES.items():
        for block in BLOCKS:
            event_dir = event_root / meta["event_dir"].format(block=block)
            event_files = sorted(event_dir.glob("events.out.tfevents.*"))
            if not event_files:
                missing.append(str(event_dir))
                continue

            event_file = event_files[-1]
            acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            acc.Reload()
            tag = meta["tag"]
            available_tags = acc.Tags().get("scalars", [])
            if tag not in available_tags:
                raise KeyError(f"Tag {tag!r} not found in {event_file}. Available tags: {available_tags}")

            for item in acc.Scalars(tag):
                rows.append(
                    {
                        "wall_time": item.wall_time,
                        "step": item.step,
                        "epoch": (item.step + 1) / STEPS_PER_EPOCH,
                        "value": item.value,
                        "tag": tag,
                        "event_file": str(event_file.relative_to(ROOT)),
                        "parameter": parameter,
                        "block": block,
                    }
                )

    if missing:
        raise FileNotFoundError("Missing TensorBoard event directories:\n" + "\n".join(missing))
    if not rows:
        raise RuntimeError(f"No gradient scalars were loaded from {event_root}")
    return pd.DataFrame(rows)


def read_gradient_data(csv_path: Path, event_root: Path, prefer_events: bool) -> tuple[pd.DataFrame, str]:
    if prefer_events or not csv_path.exists():
        df = read_tensorboard_events(event_root)
        source = f"TensorBoard events under {event_root.relative_to(ROOT)}"
    else:
        df = pd.read_csv(csv_path)
        source = f"CSV export {csv_path.relative_to(ROOT)}"
    return df, source


def validate_gradient_data(df: pd.DataFrame) -> None:
    required = {"epoch", "value", "parameter", "block"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Gradient data is missing required columns: {sorted(missing)}")

    expected = set(SERIES)
    present = set(df["parameter"].astype(str).unique())
    missing_parameters = expected.difference(present)
    if missing_parameters:
        raise ValueError(f"Missing required gradient parameter series: {sorted(missing_parameters)}")

    blocks = set(pd.to_numeric(df["block"], errors="coerce").dropna().astype(int).unique())
    missing_blocks = set(BLOCKS).difference(blocks)
    if missing_blocks:
        raise ValueError(f"Missing required representative blocks: {sorted(missing_blocks)}")

    if not (pd.to_numeric(df["value"], errors="coerce") > 0).all():
        raise ValueError("All gradient values must be positive because the figure plots log10 values.")


def prepare_plot_data(df: pd.DataFrame, smooth_span: int) -> pd.DataFrame:
    df = df.copy()
    df["parameter"] = df["parameter"].astype(str)
    df["block"] = pd.to_numeric(df["block"], errors="raise").astype(int)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="raise")
    df["value"] = pd.to_numeric(df["value"], errors="raise")
    df = df[df["parameter"].isin(SERIES) & df["block"].isin(BLOCKS)].copy()
    df = df.sort_values(["parameter", "block", "epoch"]).reset_index(drop=True)
    df["smooth_value"] = df.groupby(["parameter", "block"], group_keys=False)["value"].apply(
        lambda values: values.ewm(span=smooth_span, adjust=False).mean()
    )
    df["log10_value"] = np.log10(df["value"].to_numpy())
    df["log10_smooth_value"] = np.log10(df["smooth_value"].to_numpy())
    return df


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.72,
            "axes.edgecolor": "#8f8f8f",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#111111",
            "figure.facecolor": "#fbfaf7",
            "axes.facecolor": "#fbfaf7",
            "grid.color": "#deddd8",
            "grid.linewidth": 0.58,
            "grid.alpha": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "#fbfaf7",
            "savefig.edgecolor": "none",
        }
    )


def plot_figure(plot_df: pd.DataFrame) -> mpl.figure.Figure:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.18, 2.28), sharex=True)
    fig.subplots_adjust(left=0.066, right=0.992, top=0.78, bottom=0.235, wspace=0.28)

    legend_handles: list[mpl.lines.Line2D] = []
    for ax, (parameter, meta) in zip(axes, SERIES.items()):
        sub = plot_df[plot_df["parameter"] == parameter]
        for block in BLOCKS:
            block_df = sub[sub["block"] == block].sort_values("epoch")
            color = COLORS[block]
            ax.plot(
                block_df["epoch"],
                block_df["log10_value"],
                color=color,
                alpha=0.055,
                linewidth=0.34,
                solid_capstyle="round",
                rasterized=True,
            )
            (line,) = ax.plot(
                block_df["epoch"],
                block_df["log10_smooth_value"],
                color=color,
                linewidth=1.36,
                solid_capstyle="round",
                label=f"Block {block}",
            )
            if parameter == "attn_quantizer.log_base_alpha":
                legend_handles.append(line)

        ax.set_title(meta["title"], pad=8)
        ax.set_xlim(-1, 101)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Epoch", fontweight="bold", labelpad=3.5)
        ax.grid(True, which="major", axis="both")
        ax.tick_params(length=2.8, width=0.62, pad=2.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ylims = {
        "attn_quantizer.log_base_alpha": (-4.25, -0.65),
        "qkv.lora_B": (-2.1, 0.18),
        "mlp.fc2.lora_B": (-2.15, 0.82),
    }
    for ax, parameter in zip(axes, SERIES):
        ax.set_ylim(*ylims[parameter])
        ax.set_ylabel("")
    axes[0].set_ylabel("log10 gradient norm", fontweight="bold", labelpad=5)

    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.35,
        borderaxespad=0.0,
    )
    return fig


def save_outputs(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
    }.items():
        fig.savefig(output_dir / f"{stem}{suffix}", bbox_inches="tight", pad_inches=0.012, **kwargs)

    png_path = output_dir / f"{stem}.png"
    with Image.open(png_path) as image:
        image.save(png_path, dpi=(600, 600))


def write_caption(output_dir: Path, stem: str, source: str, smooth_span: int, n_rows: int) -> None:
    caption = (
        "Figure 6. CIFAR case-study gradient diagnostics for 3-bit QPEFT. "
        "The panels show representative gradient-norm traces for "
        "Grad_attn_quantizer.log_base_alpha, Grad_qkv.lora_B, and Grad_mlp.fc2.lora_B "
        "in blocks 0, 5, and 11, plotted as log10 gradient norms. Thin traces are raw "
        "TensorBoard scalar values and thick traces are EMA-smoothed curves with span "
        f"{smooth_span} for readability. This figure is a CIFAR case-study diagnostic "
        "rather than broad statistical validation."
    )
    metadata = (
        f"{caption}\n\n"
        f"- Data source: {source}\n"
        f"- Plotted scalar rows: {n_rows}\n"
        "- Rendering: matplotlib; no seaborn and no AI-generated curves.\n"
    )
    (output_dir / f"{stem}_caption.md").write_text(metadata, encoding="utf-8")


def write_summary(plot_df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    summary = (
        plot_df.groupby(["parameter", "block"], as_index=False)["value"]
        .agg(n="count", min="min", median="median", max="max")
        .sort_values(["parameter", "block"])
    )
    summary["gradient_name"] = summary["parameter"].map(lambda name: SERIES[name]["gradient_name"])
    summary = summary[["gradient_name", "block", "n", "min", "median", "max"]]
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    plot_df.to_csv(output_dir / f"{stem}_plotted_points.csv", index=False)


def main() -> None:
    args = parse_args()
    df, source = read_gradient_data(args.csv, args.event_root, args.prefer_events)
    validate_gradient_data(df)
    plot_df = prepare_plot_data(df, args.smooth_span)
    fig = plot_figure(plot_df)
    save_outputs(fig, args.output_dir, args.stem)
    plt.close(fig)
    write_caption(args.output_dir, args.stem, source, args.smooth_span, len(plot_df))
    write_summary(plot_df, args.output_dir, args.stem)
    print(f"Figure 6 outputs written to: {args.output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
