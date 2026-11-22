"""Save report figures from the stored results and console output.

Run from the repository root with python -m src.experiments.output_figures.
This redraws the figures without fitting or changing any model.
"""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def plot_test_ndcg(results_path: Path, output_path: Path) -> None:
    """Draw the five saved test NDCG@10 scores.

    Args:
        results_path: Comparison CSV with model and ndcg_at_10 columns.
        output_path: PNG file to save.
    """
    with results_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    values = [float(row["ndcg_at_10"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.5, 3.5))
    figure.subplots_adjust(left=0.28, right=0.89, top=0.93, bottom=0.18)
    axis.barh([row["model"] for row in rows], values, color="#6685A6", height=0.58)
    axis.invert_yaxis()
    axis.set_xlabel("Test NDCG@10", fontsize=11)
    axis.tick_params(axis="both", labelsize=10.5)
    axis.set_xlim(0, 0.062)
    axis.spines[["top", "right"]].set_visible(False)
    for position, value in enumerate(values):
        axis.text(value + 0.001, position, f"{value:.4f}", va="center", fontsize=10.5)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_hybrid_weights(results_path: Path, output_path: Path) -> None:
    """Draw the saved validation scores for the three hybrid weights.

    Args:
        results_path: CSV with itemcf_weight and validation_ndcg_at_10.
        output_path: PNG file to save.
    """
    with results_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    values = [float(row["validation_ndcg_at_10"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.5, 3.4))
    figure.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.20)
    axis.bar([row["itemcf_weight"] for row in rows], values, color="#6685A6", width=0.55)
    axis.set_xlabel("ItemCF weight", fontsize=11)
    axis.set_ylabel("Validation NDCG@10", fontsize=11)
    axis.set_ylim(0, 0.055)
    axis.tick_params(labelsize=10.5)
    axis.spines[["top", "right"]].set_visible(False)
    for position, value in enumerate(values):
        axis.text(position, value + 0.001, f"{value:.4f}", ha="center", fontsize=10.5)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_console_figure(text_path: Path, image_path: Path) -> None:
    """Display captured program output as a plain text figure.

    Args:
        text_path: UTF-8 console output recorded from recommend_cli.py.
        image_path: PNG file to save.

    The text is displayed as recorded, without adding a simulated app window.
    """
    text = text_path.read_text(encoding="utf-8").rstrip()
    line_count = len(text.splitlines())
    figure = plt.figure(figsize=(10, max(3.6, line_count * 0.25)), facecolor="white")
    figure.text(
        0.025, 0.97, text, va="top", family="DejaVu Sans Mono",
        fontsize=10.2, linespacing=1.35, color="black",
    )
    figure.savefig(image_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    """Redraw the report charts and both console figures from the saved files."""
    plot_test_ndcg(
        ROOT / "reports/comparison/test_metrics.csv",
        ROOT / "reports/comparison/ndcg_at_10.png",
    )
    plot_hybrid_weights(
        ROOT / "reports/hybrid/weight_validation.csv",
        ROOT / "reports/hybrid/weight_validation.png",
    )
    for profile in ("guest", "profile"):
        save_console_figure(
            ROOT / f"outputs/demo_{profile}.txt",
            ROOT / f"reports/hybrid/cli_{profile}.png",
        )


if __name__ == "__main__":
    main()
