from __future__ import annotations

import argparse
import math
from pathlib import Path

import altair as alt
import pandas as pd
import vl_convert as vlc
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

SERIES = {
    "attn_quantizer.log_base_alpha": {
        "tag": "Grad/attn_quantizer.log_base_alpha",
        "event_dir": "Grad_attn_quantizer.log_base_alpha_block_{block}",
        "panel": "A. Quantizer log-base alpha",
        "gradient_name": "Grad_attn_quantizer.log_base_alpha",
    },
    "qkv.lora_B": {
        "tag": "Grad/qkv.lora_B",
        "event_dir": "Grad_qkv.lora_B_block_{block}",
        "panel": "B. qkv LoRA-B",
        "gradient_name": "Grad_qkv.lora_B",
    },
    "mlp.fc2.lora_B": {
        "tag": "Grad/mlp.fc2.lora_B",
        "event_dir": "Grad_mlp.fc2.lora_B_block_{block}",
        "panel": "C. mlp.fc2 LoRA-B",
        "gradient_name": "Grad_mlp.fc2.lora_B",
    },
}
PANEL_ORDER = [item["panel"] for item in SERIES.values()]
BLOCKS = [0, 5, 11]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 6 gradient diagnostics for the CIFAR 3-bit QPEFT "
            "case study from real TensorBoard scalars or exported CSV data."
        )
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
    parser.add_argument("--smooth-span", type=positive_int, default=81, help="EMA span for the thick trend traces.")
    parser.add_argument(
        "--png-scale",
        type=float,
        default=5.0,
        help="Vega render scale for the high-resolution PNG. The PNG is tagged as 600 dpi.",
    )
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
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            tag = meta["tag"]
            available_tags = accumulator.Tags().get("scalars", [])
            if tag not in available_tags:
                raise KeyError(f"Tag {tag!r} not found in {event_file}. Available tags: {available_tags}")

            for item in accumulator.Scalars(tag):
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
        raise ValueError(f"Gradient CSV is missing required columns: {sorted(missing)}")

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
        raise ValueError("Figure 6 uses a log y-axis, so all gradient values must be positive.")


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
    df["log10_value"] = df["value"].map(math.log10)
    df["log10_smooth_value"] = df["smooth_value"].map(math.log10)
    df["panel"] = df["parameter"].map(lambda name: SERIES[name]["panel"])
    df["gradient_name"] = df["parameter"].map(lambda name: SERIES[name]["gradient_name"])
    df["block_label"] = "Block " + df["block"].astype(str)
    df["label_x"] = pd.NA
    df["label_text"] = pd.NA
    label_indices = df.groupby(["parameter", "block"])["epoch"].idxmax()
    df.loc[label_indices, "label_x"] = 101.2
    df.loc[label_indices, "label_text"] = "B" + df.loc[label_indices, "block"].astype(str)
    return df


def make_chart(plot_df: pd.DataFrame) -> alt.Chart:
    alt.data_transformers.disable_max_rows()

    color_scale = alt.Scale(
        domain=["Block 0", "Block 5", "Block 11"],
        range=["#365f91", "#c76d1d", "#3f7f4a"],
    )
    x_axis = alt.Axis(title="Epoch", values=[0, 25, 50, 75, 100], tickSize=4, labelPadding=6)
    y_axis = alt.Axis(title="log10 norm", grid=True, tickCount=5, labelPadding=5, format=".1f")
    color_no_legend = alt.Color("block_label:N", scale=color_scale, legend=None)
    base = alt.Chart(plot_df).properties(width=258, height=218).encode(
        x=alt.X("epoch:Q", title="Epoch", scale=alt.Scale(domain=[0, 104]), axis=x_axis),
        tooltip=[
            alt.Tooltip("gradient_name:N", title="Gradient"),
            alt.Tooltip("block_label:N", title="Block"),
            alt.Tooltip("epoch:Q", title="Epoch", format=".2f"),
            alt.Tooltip("value:Q", title="Raw norm", format=".3e"),
            alt.Tooltip("smooth_value:Q", title="EMA norm", format=".3e"),
        ],
    )

    raw = base.mark_line(opacity=0.04, strokeWidth=0.55).encode(
        y=alt.Y("log10_value:Q", title="log10 norm", axis=y_axis),
        color=color_no_legend,
    )
    smooth = base.mark_line(opacity=1.0, strokeWidth=2.25, strokeCap="round").encode(
        y=alt.Y("log10_smooth_value:Q", title="log10 norm", axis=y_axis),
        color=color_no_legend,
    )
    labels = base.transform_filter("isValid(datum.label_text)").mark_text(
        align="left",
        baseline="middle",
        dx=3,
        font="sans-serif",
        fontSize=9.8,
        fontWeight=700,
    ).encode(
        x=alt.X("label_x:Q", title="Epoch", scale=alt.Scale(domain=[0, 104]), axis=x_axis),
        y=alt.Y("log10_smooth_value:Q", title="log10 norm", axis=y_axis),
        text=alt.Text("label_text:N"),
        color=color_no_legend,
    )

    chart = (
        alt.layer(raw, smooth, labels)
        .facet(
            column=alt.Column(
                "panel:N",
                sort=PANEL_ORDER,
                title=None,
                header=alt.Header(
                    labelOrient="top",
                    labelAnchor="middle",
                    labelFontSize=13,
                    labelFontWeight=600,
                    labelPadding=10,
                ),
            )
        )
        .resolve_scale(y="independent")
        .properties(bounds="flush", spacing=26)
        .configure_view(stroke=None)
        .configure_axis(
            domainColor="#b8b8b8",
            domainWidth=0.8,
            gridColor="#e9e6df",
            gridOpacity=0.95,
            labelColor="#333333",
            labelFont="sans-serif",
            labelFontSize=10.5,
            titleColor="#222222",
            titleFont="sans-serif",
            titleFontSize=11.5,
            titlePadding=9,
            tickColor="#b8b8b8",
        )
        .configure_header(labelFont="sans-serif", titleFont="sans-serif")
        .configure_legend(
            labelColor="#333333",
            titleColor="#333333",
            labelFont="sans-serif",
            titleFont="sans-serif",
            padding=4,
        )
        .configure(background="#fbfaf7")
    )
    return chart


def write_outputs(chart: alt.Chart, output_dir: Path, stem: str, png_scale: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = chart.to_dict()

    svg = vlc.vegalite_to_svg(spec)
    (output_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")

    pdf = vlc.vegalite_to_pdf(spec)
    (output_dir / f"{stem}.pdf").write_bytes(pdf)

    png = vlc.vegalite_to_png(spec, scale=png_scale)
    png_path = output_dir / f"{stem}.png"
    png_path.write_bytes(png)
    with Image.open(png_path) as image:
        image.save(png_path, dpi=(600, 600))


def write_caption(output_dir: Path, stem: str, source: str, smooth_span: int, n_rows: int) -> None:
    caption = (
        "Figure 6. CIFAR case-study gradient diagnostics for 3-bit QPEFT. "
        "The panels show representative gradient-norm traces for "
        "Grad_attn_quantizer.log_base_alpha, Grad_qkv.lora_B, and Grad_mlp.fc2.lora_B "
        "in blocks 0, 5, and 11, plotted on a log10 gradient-norm scale. Thin traces are "
        "raw TensorBoard scalar values and thick "
        f"traces are EMA-smoothed curves with span {smooth_span} for readability. "
        "This figure is a CIFAR case-study diagnostic rather than broad statistical validation."
    )
    metadata = (
        f"{caption}\n\n"
        f"- Data source: {source}\n"
        f"- Plotted scalar rows: {n_rows}\n"
        "- Rendering: Altair/Vega-Lite with vl-convert; no seaborn and no AI-generated curves.\n"
    )
    (output_dir / f"{stem}_caption.md").write_text(metadata, encoding="utf-8")


def write_summary(plot_df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    summary = (
        plot_df.groupby(["gradient_name", "block"], as_index=False)["value"]
        .agg(n="count", min="min", median="median", max="max")
        .sort_values(["gradient_name", "block"])
    )
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    plot_df.to_csv(output_dir / f"{stem}_plotted_points.csv", index=False)


def main() -> None:
    args = parse_args()
    df, source = read_gradient_data(args.csv, args.event_root, args.prefer_events)
    validate_gradient_data(df)
    plot_df = prepare_plot_data(df, args.smooth_span)
    chart = make_chart(plot_df)
    write_outputs(chart, args.output_dir, args.stem, args.png_scale)
    write_caption(args.output_dir, args.stem, source, args.smooth_span, len(plot_df))
    write_summary(plot_df, args.output_dir, args.stem)
    print(f"Figure 6 outputs written to: {args.output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
