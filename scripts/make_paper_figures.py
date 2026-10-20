"""Generate compact, publication-ready figures for the revised manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "artifacts"


def _box(ax, y: float, title: str, detail: str, color: str) -> None:
    patch = FancyBboxPatch(
        (0.08, y), 0.84, 0.13,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=0.8, edgecolor="#263238", facecolor=color,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(0.5, y + 0.088, title, ha="center", va="center", fontsize=8.2, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, y + 0.043, detail, ha="center", va="center", fontsize=5.8, transform=ax.transAxes)


def protocol_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.25, 5.1), dpi=300)
    ax.set_axis_off()
    items = [
        (0.79, "Publication-eligible email sources", "Checksums, label ontology, license, provenance", "#D8EAF3"),
        (0.60, "Source-aware preprocessing", "Mask volatile IOCs; exact + template hashes", "#E8E3F3"),
        (0.41, "Four group-isolated roles", "ML train | fusion train | threshold | locked test", "#DDEEDB"),
        (0.22, "Frozen matched Local-LLM cohort", "5,000 template-unique cases; digest-pinned Bonsai", "#FCE8CF"),
        (0.03, "Confirmatory comparison", "Seven configurations; paired group-bootstrap ΔF1", "#F6D6D6"),
    ]
    for y, title, detail, color in items:
        _box(ax, y, title, detail, color)
    for upper, lower in zip(items, items[1:]):
        ax.annotate(
            "", xy=(0.5, lower[0] + 0.145), xytext=(0.5, upper[0] - 0.01),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#455A64"},
        )
    ax.text(0.5, 0.975, "Leakage-resistant paper protocol", ha="center", va="top", fontsize=9.2, fontweight="bold", transform=ax.transAxes)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def delta_figure(summary: dict, output: Path) -> None:
    labels = {
        "Technical_Matrix_only": "Matrix only",
        "LLM_only": "Local LLM only",
        "ML_plus_Matrix": "ML + matrix",
        "ML_plus_LLM": "ML + Local LLM",
        "Matrix_plus_LLM": "Matrix + Local LLM",
        "Full_ML_Matrix_LLM": "Full hybrid",
    }
    rows = [
        row for row in summary["paired_deltas"]
        if row["dataset"] == "locked_internal_test"
        and row["variant"] == "all"
        and row["candidate"] in labels
    ]
    rows.sort(key=lambda row: list(labels).index(row["candidate"]))
    points = [100 * row["point_delta_candidate_minus_reference"]["f1"] for row in rows]
    intervals = [row["paired_grouped_bootstrap_95_ci"]["intervals"]["f1"] for row in rows]
    low = [100 * item[0] for item in intervals]
    high = [100 * item[1] for item in intervals]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(3.25, 3.15), dpi=300)
    ax.axvline(0, color="#455A64", linewidth=0.8, linestyle="--")
    for index, (point, lower, upper) in enumerate(zip(points, low, high)):
        color = "#0072B2" if point >= 0 else "#D55E00"
        ax.errorbar(
            point, index,
            xerr=[[point - lower], [upper - point]],
            fmt="o", color=color, ecolor=color,
            capsize=2.5, markersize=4, linewidth=1.0,
        )
    ax.set_yticks(y, [labels[row["candidate"]] for row in rows], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("ΔF1 vs. ML only (percentage points, 95% CI)", fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Paired template-group bootstrap", fontsize=8.6, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create revised paper figures")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_figure(args.output / "figure1_protocol.png")
    if args.summary:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        if not summary.get("paper_gate", {}).get("passed"):
            raise ValueError("Refusing to plot performance before the paper gate passes.")
        delta_figure(summary, args.output / "figure2_paired_f1_delta.png")
    print(json.dumps({"output": str(args.output.resolve())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
