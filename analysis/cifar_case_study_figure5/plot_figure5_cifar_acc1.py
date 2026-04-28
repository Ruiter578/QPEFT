from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "analysis/tensorboard_cifar_3bit_qpeft/csv/val_acc1.csv"
DEFAULT_EVENT = (
    ROOT
    / "outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/20260114184203_default_acc=53.50"
    / "tensorboard/events.out.tfevents.1768387332.master-192-168-8-48.1680.0"
)
DEFAULT_TENSORBOARD_SCREENSHOT = ROOT / "tensorboard.png"
DEFAULT_OUT_DIR = ROOT / "analysis/cifar_case_study_figure5/figures"
FIGURE_STEM = "figure5_cifar_3bit_qpeft_val_acc1_trajectory"


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.78,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.pad_inches": 0.03,
        }
    )


def read_csv_curve(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scalar CSV not found: {path}")
    df = pd.read_csv(path)
    if "epoch" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{path} must contain 'epoch' and 'value' columns.")
    df = df[["epoch", "value"]].dropna().sort_values("epoch").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path} contains no valid rows.")
    return df


def read_tensorboard_curve(event_file: Path, tag: str) -> pd.DataFrame:
    if not event_file.exists():
        raise FileNotFoundError(f"TensorBoard event file not found: {event_file}")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise ImportError("tensorboard is required for --event-file input. Use --csv if unavailable.") from exc

    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(f"Tag {tag!r} not found in {event_file}. Available scalar tags: {tags}")
    rows = [{"epoch": item.step, "value": item.value} for item in acc.Scalars(tag)]
    df = pd.DataFrame(rows).dropna().sort_values("epoch").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Tag {tag!r} in {event_file} contains no scalar rows.")
    return df


def save_multi_format(fig: mpl.figure.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")


def plot_curve(df: pd.DataFrame, out_dir: Path, warmup_epochs: int, rise_end: int) -> dict[str, float]:
    set_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df["epoch"].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)
    final_epoch, final_acc = x[-1], y[-1]
    best_idx = int(np.argmax(y))
    best_epoch, best_acc = x[best_idx], y[best_idx]

    fig, ax = plt.subplots(figsize=(6.45, 3.05), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")

    ax.set_xlim(8, 101)
    ax.set_ylim(0, max(58, best_acc + 4.0))
    ax.axvspan(0, warmup_epochs, color="#DCEBFA", alpha=0.34, lw=0, zorder=0)
    ax.axvspan(warmup_epochs, rise_end, color="#FFF4CF", alpha=0.34, lw=0, zorder=0)
    ax.axhline(best_acc, color="#C73743", lw=0.78, ls=(0, (4, 4)), alpha=0.56, zorder=1)

    ax.fill_between(x, y, 0, color="#2A65A0", alpha=0.060, zorder=1)

    ax.plot(
        x,
        y,
        color="#1E5B97",
        lw=2.45,
        marker="o",
        ms=4.7,
        mfc="white",
        mec="#1E5B97",
        mew=1.35,
        zorder=4,
    )
    ax.scatter([best_epoch], [best_acc], s=175, color="#D94A56", alpha=0.18, edgecolor="none", zorder=5)
    ax.scatter([best_epoch], [best_acc], s=64, color="#C73743", edgecolor="white", linewidth=1.05, zorder=6)
    ax.annotate(
        f"Best / final Acc@1: {best_acc:.1f}%\nepoch {int(best_epoch)}",
        xy=(best_epoch, best_acc),
        xytext=(61.0, 39.0),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#4B5563", "shrinkA": 3, "shrinkB": 5},
        color="#1F2933",
        fontsize=8.35,
        bbox={"boxstyle": "round,pad=0.34", "fc": "white", "ec": "#CBD5DF", "lw": 0.7, "alpha": 0.96},
        zorder=7,
    )
    ax.annotate(
        f"Start: {y[0]:.1f}%",
        xy=(x[0], y[0]),
        xytext=(17.5, 9.0),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#6B7280", "shrinkA": 3, "shrinkB": 5},
        color="#4B5563",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.26", "fc": "white", "ec": "#E3E8EF", "lw": 0.55, "alpha": 0.92},
        zorder=7,
    )

    ax.set_title("CIFAR Case Study: 3-bit QPEFT Validation Acc@1", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Acc@1 (%)")
    ax.set_xticks([10, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.grid(True, axis="y", color="#D6DCE3", linewidth=0.65, alpha=0.66)
    ax.grid(True, axis="x", color="#EEF1F5", linewidth=0.55, alpha=0.42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4B5563")
    ax.spines["bottom"].set_color("#4B5563")

    save_multi_format(fig, out_dir, FIGURE_STEM)
    plt.close(fig)

    summary_df = pd.DataFrame(
        [
            {"metric": "first_logged_acc1", "epoch": x[0], "value": y[0]},
            {"metric": "warmup_boundary_acc1_interp", "epoch": warmup_epochs, "value": float(np.interp(warmup_epochs, x, y))},
            {"metric": "best_acc1", "epoch": best_epoch, "value": best_acc},
            {"metric": "final_acc1", "epoch": final_epoch, "value": final_acc},
        ]
    )
    summary_df.to_csv(out_dir / f"{FIGURE_STEM}_summary.csv", index=False)
    summary_df.to_csv(out_dir / "figure5_cifar_3bit_qpeft_val_acc1_summary.csv", index=False)
    return {
        "first_epoch": float(x[0]),
        "first_acc1": float(y[0]),
        "best_epoch": float(best_epoch),
        "best_acc1": float(best_acc),
        "final_epoch": float(final_epoch),
        "final_acc1": float(final_acc),
    }


def write_metadata(
    out_dir: Path,
    source_type: str,
    source_path: Path,
    tag: str,
    tensorboard_screenshot: Path,
    stats: dict[str, float],
) -> None:
    metadata = {
        "figure": "Figure 5",
        "caption": "CIFAR case-study training trajectory for 3-bit QPEFT.",
        "source_type": source_type,
        "source_path": str(source_path),
        "tensorboard_screenshot_reference": str(tensorboard_screenshot),
        "tensorboard_tag": tag,
        "note": "This figure is a CIFAR case-study trajectory and does not claim multi-task convergence.",
        "stats": stats,
    }
    (out_dir / f"{FIGURE_STEM}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 5 CIFAR 3-bit QPEFT validation Acc@1 trajectory.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV with epoch and value columns.")
    parser.add_argument("--event-file", type=Path, default=None, help="TensorBoard event file fallback/source.")
    parser.add_argument(
        "--tensorboard-screenshot",
        type=Path,
        default=DEFAULT_TENSORBOARD_SCREENSHOT,
        help="Optional TensorBoard screenshot reference. It is not used as numeric data.",
    )
    parser.add_argument("--tag", default="Val/epoch_acc1", help="Scalar tag to read from TensorBoard event files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for figure files.")
    parser.add_argument("--warmup-epochs", type=int, default=20, help="Warmup boundary in epochs.")
    parser.add_argument("--rise-end", type=int, default=40, help="Boundary between rapid adaptation and stabilization.")
    parser.add_argument("--prefer-event", action="store_true", help="Read TensorBoard event file even if CSV exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefer_event or not args.csv.exists():
        event_file = args.event_file or DEFAULT_EVENT
        df = read_tensorboard_curve(event_file, args.tag)
        source_type = "tensorboard_event"
        source_path = event_file
    else:
        df = read_csv_curve(args.csv)
        source_type = "scalar_csv"
        source_path = args.csv
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / f"{FIGURE_STEM}_points.csv", index=False)
    stats = plot_curve(df, args.out_dir, args.warmup_epochs, args.rise_end)
    write_metadata(args.out_dir, source_type, source_path, args.tag, args.tensorboard_screenshot, stats)
    print(f"Figure 5 outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
