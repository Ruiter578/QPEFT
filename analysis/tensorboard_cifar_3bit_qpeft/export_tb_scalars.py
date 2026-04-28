from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/20260114184203_default_acc=53.50"
OUT_DIR = Path(__file__).resolve().parent
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"

STEPS_PER_EPOCH = 32.0

MAIN_EVENT = RUN_DIR / "tensorboard/events.out.tfevents.1768387332.master-192-168-8-48.1680.0"

MAIN_TAGS = {
    "val_acc1": "Val/epoch_acc1",
    "val_acc5": "Val/epoch_acc5",
    "val_loss": "Val/epoch_loss",
    "train_loss": "Train/iter_loss",
    "learning_rate": "Train/iter_lr",
    "drop_prob_mean": "Train/mean_prob",
    "drop_prob_min": "Train/min_prob",
    "drop_prob_max": "Train/max_prob",
}

GRADIENT_SERIES = [
    (
        "attn_quantizer.log_base_alpha",
        "Grad/attn_quantizer.log_base_alpha",
        {
            0: RUN_DIR / "tensorboard/Grad_attn_quantizer.log_base_alpha_block_0/events.out.tfevents.1768387347.master-192-168-8-48.1680.5",
            5: RUN_DIR / "tensorboard/Grad_attn_quantizer.log_base_alpha_block_5/events.out.tfevents.1768387348.master-192-168-8-48.1680.6",
            11: RUN_DIR / "tensorboard/Grad_attn_quantizer.log_base_alpha_block_11/events.out.tfevents.1768387348.master-192-168-8-48.1680.7",
        },
    ),
    (
        "qkv.lora_B",
        "Grad/qkv.lora_B",
        {
            0: RUN_DIR / "tensorboard/Grad_qkv.lora_B_block_0/events.out.tfevents.1768387346.master-192-168-8-48.1680.2",
            5: RUN_DIR / "tensorboard/Grad_qkv.lora_B_block_5/events.out.tfevents.1768387346.master-192-168-8-48.1680.3",
            11: RUN_DIR / "tensorboard/Grad_qkv.lora_B_block_11/events.out.tfevents.1768387346.master-192-168-8-48.1680.4",
        },
    ),
    (
        "mlp.fc2.lora_B",
        "Grad/mlp.fc2.lora_B",
        {
            0: RUN_DIR / "tensorboard/Grad_mlp.fc2.lora_B_block_0/events.out.tfevents.1768387349.master-192-168-8-48.1680.8",
            5: RUN_DIR / "tensorboard/Grad_mlp.fc2.lora_B_block_5/events.out.tfevents.1768387349.master-192-168-8-48.1680.9",
            11: RUN_DIR / "tensorboard/Grad_mlp.fc2.lora_B_block_11/events.out.tfevents.1768387349.master-192-168-8-48.1680.10",
        },
    ),
]


def load_scalars(event_path: Path, tag: str) -> pd.DataFrame:
    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(f"Tag {tag!r} not found in {event_path}. Available tags: {tags}")
    rows = [
        {
            "wall_time": item.wall_time,
            "step": item.step,
            "epoch": item.step,
            "value": item.value,
            "tag": tag,
            "event_file": str(event_path.relative_to(ROOT)),
        }
        for item in acc.Scalars(tag)
    ]
    return pd.DataFrame(rows)


def save_multi_format(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 600},
    }.items():
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kwargs)


def smooth(values: pd.Series, span: int = 81) -> pd.Series:
    return values.ewm(span=span, adjust=False).mean()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
        }
    )


def export_main_curves() -> dict[str, pd.DataFrame]:
    curves = {}
    for name, tag in MAIN_TAGS.items():
        df = load_scalars(MAIN_EVENT, tag)
        if name in {"train_loss", "learning_rate"}:
            df["epoch"] = df["step"] / STEPS_PER_EPOCH
        else:
            df["epoch"] = df["step"]
        csv_path = CSV_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        curves[name] = df
    return curves


def export_gradient_curves() -> pd.DataFrame:
    frames = []
    for param_name, tag, block_to_event in GRADIENT_SERIES:
        for block, event_path in block_to_event.items():
            df = load_scalars(event_path, tag)
            df["parameter"] = param_name
            df["block"] = block
            df["epoch"] = (df["step"] + 1) / STEPS_PER_EPOCH
            df.to_csv(CSV_DIR / f"grad_{param_name.replace('.', '_')}_block_{block}.csv", index=False)
            frames.append(df)
    all_grads = pd.concat(frames, ignore_index=True)
    all_grads.to_csv(CSV_DIR / "gradients_selected_long.csv", index=False)
    return all_grads


def plot_training_overview(curves: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05), constrained_layout=True)

    ax = axes[0]
    ax.plot(curves["val_acc1"]["step"], curves["val_acc1"]["value"], marker="o", ms=3.3, label="Acc@1")
    ax.plot(curves["val_acc5"]["step"], curves["val_acc5"]["value"], marker="s", ms=3.1, label="Acc@5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    train = curves["train_loss"].copy()
    ax.plot(train["epoch"], train["value"], color="#4C78A8", alpha=0.18, linewidth=0.7)
    ax.plot(train["epoch"], smooth(train["value"]), color="#1F4E79", label="EMA")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("Optimization")
    ax.grid(True, alpha=0.22, linewidth=0.6)

    ax = axes[2]
    lr = curves["learning_rate"]
    ax.plot(lr["epoch"], lr["value"], color="#D55E00")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Schedule")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    save_multi_format(fig, "training_overview")
    plt.close(fig)


def plot_gradients(all_grads: pd.DataFrame) -> None:
    colors = {0: "#4C78A8", 5: "#F58518", 11: "#54A24B"}
    titles = {
        "attn_quantizer.log_base_alpha": "Attention log-base gradient",
        "qkv.lora_B": "QKV LoRA-B gradient",
        "mlp.fc2.lora_B": "MLP fc2 LoRA-B gradient",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), constrained_layout=True, sharex=True)
    for ax, param_name in zip(axes, titles):
        sub = all_grads[all_grads["parameter"] == param_name]
        for block, block_df in sub.groupby("block"):
            block_df = block_df.sort_values("epoch")
            ax.plot(block_df["epoch"], block_df["value"], color=colors[int(block)], alpha=0.10, linewidth=0.55)
            ax.plot(
                block_df["epoch"],
                smooth(block_df["value"]),
                color=colors[int(block)],
                label=f"Block {int(block)}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(titles[param_name])
        ax.grid(True, which="both", alpha=0.20, linewidth=0.55)
    axes[0].set_ylabel("Gradient norm")
    axes[-1].legend(frameon=False, loc="upper right")
    save_multi_format(fig, "selected_gradients")
    plt.close(fig)


def plot_gradient_panels(all_grads: pd.DataFrame) -> None:
    colors = {0: "#4C78A8", 5: "#F58518", 11: "#54A24B"}
    for param_name, sub in all_grads.groupby("parameter"):
        fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
        for block, block_df in sub.groupby("block"):
            block_df = block_df.sort_values("epoch")
            ax.plot(block_df["epoch"], block_df["value"], color=colors[int(block)], alpha=0.10, linewidth=0.55)
            ax.plot(block_df["epoch"], smooth(block_df["value"]), color=colors[int(block)], label=f"Block {int(block)}")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient norm")
        ax.set_title(param_name)
        ax.grid(True, which="both", alpha=0.20, linewidth=0.55)
        ax.legend(frameon=False, loc="best")
        stem = "gradient_" + param_name.replace(".", "_")
        save_multi_format(fig, stem)
        plt.close(fig)


def export_summary(curves: dict[str, pd.DataFrame], all_grads: pd.DataFrame) -> None:
    rows = []
    for name, df in curves.items():
        rows.append(
            {
                "series": name,
                "n": len(df),
                "first_step": int(df["step"].iloc[0]),
                "last_step": int(df["step"].iloc[-1]),
                "first_value": float(df["value"].iloc[0]),
                "last_value": float(df["value"].iloc[-1]),
                "min_value": float(df["value"].min()),
                "max_value": float(df["value"].max()),
            }
        )
    for (param, block), df in all_grads.groupby(["parameter", "block"]):
        rows.append(
            {
                "series": f"grad/{param}/block_{block}",
                "n": len(df),
                "first_step": int(df["step"].iloc[0]),
                "last_step": int(df["step"].iloc[-1]),
                "first_value": float(df["value"].iloc[0]),
                "last_value": float(df["value"].iloc[-1]),
                "min_value": float(df["value"].min()),
                "max_value": float(df["value"].max()),
            }
        )
    pd.DataFrame(rows).to_csv(CSV_DIR / "export_summary.csv", index=False)


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    curves = export_main_curves()
    all_grads = export_gradient_curves()
    export_summary(curves, all_grads)
    plot_training_overview(curves)
    plot_gradients(all_grads)
    plot_gradient_panels(all_grads)

    print(f"Run directory: {RUN_DIR.relative_to(ROOT)}")
    print(f"CSV directory: {CSV_DIR.relative_to(ROOT)}")
    print(f"Figure directory: {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
