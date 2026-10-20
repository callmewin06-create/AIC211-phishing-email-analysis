"""Publication-style figures for the benchmark and ablation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]


def _latest(root: Path) -> Path:
    pointer = json.loads((root / "latest_benchmark.json").read_text(encoding="utf-8"))
    path = Path(pointer["path"])
    return path if path.is_absolute() else ROOT / path


def _style():
    sns.set_theme(style="white", context="paper", palette="colorblind")
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def generate(run_dir: str | Path | None = None) -> dict:
    root = Path(run_dir) if run_dir else _latest(ROOT / "research_results")
    if not root.is_absolute():
        root = ROOT / root
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = {}

    benchmark = pd.read_csv(root / "benchmark_metrics.csv")
    external = benchmark.loc[
        benchmark["variant"].eq("unique_template") | benchmark["dataset"].eq("internal_calibration")
    ].copy()
    external["dataset_label"] = external["dataset"].replace({
        "internal_calibration": "Internal calibration",
        "spaphish_v5": "SpaPhish v5",
        "uot_validation_2024": "UoT 2024 (unique templates)",
    })
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    sns.pointplot(
        data=external,
        x="dataset_label",
        y="average_precision",
        hue="model",
        dodge=0.35,
        markers="o",
        linestyles="",
        errorbar=None,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Average precision (PR-AUC)")
    ax.set_title("Cross-dataset benchmark with source-aware preprocessing")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Model", frameon=False, ncol=2, loc="lower left")
    path = out / "benchmark_pr_auc.pdf"
    fig.savefig(path)
    fig.savefig(out / "benchmark_pr_auc.png", dpi=300)
    plt.close(fig)
    outputs["benchmark_pr_auc"] = str(path)

    model_order = [
        "logistic_regression",
        "linear_svm",
        "multinomial_nb",
        "random_forest",
        "xgboost",
    ]
    model_labels = {
        "logistic_regression": "LogReg",
        "linear_svm": "Linear SVM",
        "multinomial_nb": "Multinomial NB",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    comparison = external.loc[external["model"].isin(model_order)].copy()
    comparison["model_label"] = pd.Categorical(
        comparison["model"].map(model_labels),
        categories=[model_labels[name] for name in model_order],
        ordered=True,
    )
    heatmap_data = comparison.pivot_table(
        index="model_label",
        columns="dataset_label",
        values="average_precision",
        observed=False,
    )
    heatmap_data = heatmap_data.reindex([model_labels[name] for name in model_order])
    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)
    sns.heatmap(
        heatmap_data,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Average precision (PR-AUC)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Baseline comparison across all trained models")
    path = out / "model_comparison_pr_auc.pdf"
    fig.savefig(path)
    fig.savefig(out / "model_comparison_pr_auc.png", dpi=300)
    plt.close(fig)
    outputs["model_comparison_pr_auc"] = str(path)

    ablation_path = root / "ablation_metrics.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        ablation = ablation.loc[ablation["variant"].eq("unique_template") | ablation["dataset"].eq("internal_ablation_holdout")].copy()
        ablation["dataset_label"] = ablation["dataset"].replace({
            "internal_ablation_holdout": "Internal holdout",
            "spaphish_v5": "SpaPhish v5",
            "uot_validation_2024": "UoT 2024 (unique templates)",
        })
        fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
        sns.pointplot(
            data=ablation,
            x="dataset_label",
            y="average_precision",
            hue="configuration",
            dodge=0.45,
            markers="o",
            linestyles="",
            errorbar=None,
            ax=ax,
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        ax.set_ylabel("Average precision (PR-AUC)")
        ax.set_title("Ablation: technical matrix contribution is dataset-dependent")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Configuration", frameon=False, fontsize=6, ncol=2, loc="lower left")
        path = out / "ablation_pr_auc.pdf"
        fig.savefig(path)
        fig.savefig(out / "ablation_pr_auc.png", dpi=300)
        plt.close(fig)
        outputs["ablation_pr_auc"] = str(path)

    pilot_path = root / "llm_pilot_ablation_metrics.csv"
    if pilot_path.exists():
        pilot = pd.read_csv(pilot_path)
        pilot = pilot.loc[
            pilot["variant"].eq("pilot_unique_template")
            | pilot["dataset"].eq("internal_ablation_holdout")
        ].copy()
        pilot["dataset_label"] = pilot["dataset"].replace({
            "internal_ablation_holdout": "Internal holdout\n(n=50)",
            "spaphish_v5": "SpaPhish v5\n(n=20)",
            "uot_validation_2024": "UoT unique templates\n(n varies)",
        })
        fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        sns.pointplot(
            data=pilot,
            x="dataset_label",
            y="average_precision",
            hue="configuration",
            dodge=0.5,
            markers="o",
            linestyles="",
            errorbar=None,
            ax=ax,
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        ax.set_ylabel("Average precision (PR-AUC)")
        ax.set_title("Exploratory matched-subset Local-LLM ablation (not confirmatory)")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Configuration", frameon=False, fontsize=6, ncol=2, loc="lower left")
        path = out / "llm_pilot_ablation_pr_auc.pdf"
        fig.savefig(path)
        fig.savefig(out / "llm_pilot_ablation_pr_auc.png", dpi=300)
        plt.close(fig)
        outputs["llm_pilot_ablation_pr_auc"] = str(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate publication-style benchmark figures.")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    print(json.dumps(generate(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
