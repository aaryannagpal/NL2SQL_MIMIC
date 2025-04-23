import pandas as pd
import numpy as np
import sys, os
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path().resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from utils.query_evaluator import QueryEvaluator
from config import PROCESSED_RESULT_DIR, STORE_ANALYSIS_DIR

evaluator = QueryEvaluator()

test = "zeroshot"
model_type = "m1"

results_path = PROCESSED_RESULT_DIR / model_type / test
analysis_path = STORE_ANALYSIS_DIR / model_type / test

analysis_data = []
for i in tqdm(os.listdir(results_path), desc="Processing files"):
    if i.endswith(".csv.gz"):
        df = pd.read_csv(results_path / i, compression="gzip")
    else:
        df = pd.read_csv(results_path / i)
    temp = evaluator.compare_queries(df)
    temp["model_name"] = i.split(".csv")[0].replace("_results", "").replace("_", " ")
    analysis_data.append(temp)

analysis_df = pd.concat(analysis_data, ignore_index=True)
analysis_df = analysis_df[
    [
        "id",
        "question",
        "true_query",
        "gen_query",
        "null_query",
        "true_tables",
        "gen_tables",
        "tables_union",
        "tables_intersection",
        "table_access_accuracy",
        "true_columns",
        "gen_columns",
        "columns_union",
        "columns_intersection",
        "column_access_accuracy",
        "query_similarity",
        "execution_plan_available",
        "execution_plan_similarity",
        "results_match",
        "result_comparison",
        "model_name",
    ]
]

analysis_df.to_csv(f"./{test}_analysis.csv", index=False)

overall_scores = (
    analysis_df.groupby(["model_name"])
    .agg(
        {
            "table_access_accuracy": "mean",
            "column_access_accuracy": "mean",
            "query_similarity": "mean",
            "results_match": "mean",
        }
    )
    .sort_values("table_access_accuracy", ascending=False)
    .reset_index()
)

overall_scores["null_query"] = "overall"

sep_scores = (
    analysis_df.groupby(["model_name", "null_query"])
    .agg(
        {
            "table_access_accuracy": "mean",
            "column_access_accuracy": "mean",
            "query_similarity": "mean",
            "results_match": "mean",
        }
    )
    .sort_values("table_access_accuracy", ascending=False)
    .reset_index()
)

scores = pd.concat([overall_scores, sep_scores], ignore_index=True)
scores.to_csv(f"./{test}_scores.csv")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi, ceil


def radar_plot(df, test, null_filter=None, group_chart=False, output_file=None):
    sns.set_style("whitegrid")
    plot_df = df[df["null_query"] == null_filter] if null_filter is not None else df

    metrics = [
        "table_access_accuracy",
        "column_access_accuracy",
        "query_similarity",
        "results_match",
    ]
    labels = ["Table Accuracy", "Column Accuracy", "Query Similarity", "Results Match"]
    angles = np.linspace(0, 2 * pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    models = plot_df["model_name"].unique()

    if group_chart:
        cols = min(3, len(models))
        rows = ceil(len(models) / cols)
        fig = plt.figure(figsize=(5 * cols, 5 * rows))

        for i, model in enumerate(models):
            ax = plt.subplot(rows, cols, i + 1, polar=True)

            ax.set_theta_offset(pi / 2)  # Start at top
            ax.set_theta_direction(-1)  # Go clockwise
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.set_facecolor("#f8f9fa")

            model_df = plot_df[plot_df["model_name"] == model]

            for _, row in model_df.iterrows():
                null_query = row["null_query"]

                values = [row[m] for m in metrics]
                values += values[:1]

                color = "#4285F4" if null_query == 0 else "#DB4437"
                style = "solid" if null_query == 0 else "dashed"
                label = "Valid" if null_query == 0 else "Null"

                ax.plot(
                    angles,
                    values,
                    linewidth=2,
                    linestyle=style,
                    color=color,
                    label=label,
                )
                ax.fill(angles, values, color=color, alpha=0.1)

            ax.set_title(model)

            if len(model_df) > 1:
                ax.legend(loc="upper right", fontsize=8)

        title = f"{test.capitalize()} Model Performance ({model_type.capitalize()})"
        if null_filter == 0:
            title += " - Valid Queries"
        elif null_filter == 1:
            title += " - Null Queries"
        else:
            title += " - All Queries"

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

        ax.set_theta_offset(pi / 2)  # Start at top
        ax.set_theta_direction(-1)  # Go clockwise
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)

        ax.grid(alpha=0.3)
        ax.set_facecolor("#f8f9fa")

        colors = sns.color_palette("viridis", len(models))
        color_map = {model: colors[i] for i, model in enumerate(models)}
        styles = {0: "solid", 1: "dashed"}

        for _, row in plot_df.iterrows():
            model = row["model_name"]
            null_query = row["null_query"]

            values = [row[m] for m in metrics]
            values += values[:1]  # Close loop

            color = color_map[model]
            style = styles.get(null_query, "solid")
            label = (
                f"{model}"
                if null_filter is not None
                else f"{model} ({'Null' if null_query else 'Valid'})"
            )

            ax.plot(
                angles, values, linewidth=2.5, linestyle=style, color=color, label=label
            )
            ax.fill(angles, values, color=color, alpha=0.1)

        title = f"{test.capitalize()} Model Performance"
        if null_filter == 0:
            title += " - Valid Queries"
        elif null_filter == 1:
            title += " - Null Queries"

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


output_map = {1: "null", 0: "not_null", "overall": "all"}
for i in [0, 1, "overall"]:
    radar_plot(
        scores,
        test,
        null_filter=i,
        group_chart=True,
        output_file=f"./{output_map[i]}_grouped_radar",
    )


def bar_chart(df, test, metric="results_match", output_file=None):
    sns.set_style("whitegrid")
    models = df["model_name"].unique()

    data = []
    for model in models:
        valid = df[(df["model_name"] == model) & (df["null_query"] == 0)][metric].values
        null = df[(df["model_name"] == model) & (df["null_query"] == 1)][metric].values
        all_rows = df[(df["model_name"] == model) & (df["null_query"] == "overall")][
            metric
        ].values

        data.append(
            {
                "model": model,
                "Valid": valid[0] if len(valid) > 0 else 0,
                "Null": null[0] if len(null) > 0 else 0,
                "Overall": all_rows[0] if len(all_rows) > 0 else 0,
            }
        )

    fig, ax = plt.subplots(figsize=(15, 6))

    width = 0.25
    x = np.arange(len(models))
    colors = ["#4285F4", "#DB4437", "#0F9D58"]  # Blue, Red, Green

    for i, col in enumerate(["Valid", "Null", "Overall"]):
        pos = x + i * width
        bars = ax.bar(
            pos,
            [d[col] for d in data],
            width=width,
            color=colors[i],
            label=col,
            alpha=0.85,
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + width / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

    metric_name = " ".join(word.capitalize() for word in metric.split("_"))
    ax.set_title(
        f"{test.capitalize()} {metric_name} by Model ({model_type.capitalize()})",
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig, ax


for metric in [
    "table_access_accuracy",
    "column_access_accuracy",
    "query_similarity",
    "results_match",
]:
    bar_chart(scores, test, metric=metric, output_file=f"./{metric}_bar.png")
