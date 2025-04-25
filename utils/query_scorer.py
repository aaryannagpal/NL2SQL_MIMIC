import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi, ceil
import sys, os
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.query_evaluator import QueryEvaluator
from config import PROCESSED_RESULT_DIR, STORE_ANALYSIS_DIR

class M1Scoring:
    def __init__(self, test, model_type, results_path=None, analysis_path=None):
        """
        Initialize M1Scoring with paths and settings
        
        Args:
            results_path: Path to processed results (default: from config)
            analysis_path: Path to store analysis (default: from config)
            test: Test name (default: "schema_aware")
            model_type: Model type (default: "m1")
        """
        self.test = test
        self.model_type = model_type
        self.results_path = results_path or PROCESSED_RESULT_DIR / model_type / test
        self.analysis_path = analysis_path or STORE_ANALYSIS_DIR / model_type / test
        self.evaluator = QueryEvaluator()
        self.analysis_df = None
        self.scores = None

    def process_results(self):
        """Process result files and create analysis dataframe"""
        analysis_data = []
        for i in tqdm(os.listdir(self.results_path), desc="Processing files"):
            if i.endswith(".csv.gz"):
                df = pd.read_csv(self.results_path / i, compression="gzip")
            else:
                df = pd.read_csv(self.results_path / i)
            temp = self.evaluator.compare_queries(df)
            temp["model_name"] = i.split(".csv")[0].replace("_results", "").replace("_", " ")
            analysis_data.append(temp)

        self.analysis_df = pd.concat(analysis_data, ignore_index=True)
        self.analysis_df = self.analysis_df[
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
        return self.analysis_df

    def calculate_scores(self):
        """Calculate overall and separate scores"""
        if self.analysis_df is None:
            raise ValueError("Analysis dataframe not found. Run process_results() first.")

        # Calculate overall scores
        overall_scores = (
            self.analysis_df.groupby(["model_name"])
            .agg(
                {
                    "table_access_accuracy": "mean",
                    "column_access_accuracy": "mean",
                    "query_similarity": "mean",
                    "results_match": "mean",
                }
            )
            .sort_values("results_match", ascending=False)  # Sort by results match (most important)
            .reset_index()
        )
        overall_scores["null_query"] = "overall"

        # Calculate separate scores by null_query
        sep_scores = (
            self.analysis_df.groupby(["model_name", "null_query"])
            .agg(
                {
                    "table_access_accuracy": "mean",
                    "column_access_accuracy": "mean",
                    "query_similarity": "mean",
                    "results_match": "mean",
                }
            )
            .sort_values("results_match", ascending=False)
            .reset_index()
        )

        # Combine all scores
        self.scores = pd.concat([overall_scores, sep_scores], ignore_index=True)
        return self.scores

    def calculate_score(self, df):
        """
        Calculate weighted scores with emphasis on results_match
        
        Weights:
        - results_match: 40% (most important)
        - table_access_accuracy: 20%
        - column_access_accuracy: 20%
        - query_similarity: 20%
        """
        weights = {
            "results_match": 0.40,         # Highest importance
            "table_access_accuracy": 0.20,
            "column_access_accuracy": 0.20,
            "query_similarity": 0.20
        }
        
        result = df.copy()
        
        # Calculate weighted score
        weighted_sum = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in result.columns:
                result[f"{metric}_weighted"] = result[metric] * weight
                weighted_sum += result[f"{metric}_weighted"]
                total_weight += weight
        
        # Normalize by available weight
        if total_weight > 0:
            result["weighted_score"] = (weighted_sum / total_weight).round(2)
        else:
            result["weighted_score"] = 0.0
        
        return result

    def radar_plot(self, df, null_filter=None, group_chart=False, output_file=None):
        """Create radar chart with weighted scoring for SQL metrics comparison"""
        # Setup
        sns.set_style("whitegrid")
        plot_df = df[df["null_query"] == null_filter] if null_filter is not None else df
        
        # Calculate scores
        scored_df = self.calculate_score(plot_df)
        
        # Chart setup
        metrics = ["table_access_accuracy", "column_access_accuracy", "query_similarity", "results_match"]
        labels = ["Table Accuracy", "Column Accuracy", "Query Similarity", "Results Match"]
        angles = np.linspace(0, 2*pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close loop
        
        # Get unique models
        models = plot_df["model_name"].unique()
        
        if group_chart:
            # Create subplot grid
            cols = min(3, len(models))
            rows = ceil(len(models) / cols)
            fig = plt.figure(figsize=(4.5 * cols, 4 * rows))
            
            # Plot each model separately
            for i, model in enumerate(models):
                # Create subplot
                ax = plt.subplot(rows, cols, i + 1, polar=True)
                
                # Configure axes
                ax.set_theta_offset(pi/2)  # Start at top
                ax.set_theta_direction(-1)  # Go clockwise
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylim(0, 1)
                ax.grid(alpha=0.3)
                ax.set_facecolor("#f8f9fa")
                
                # Get data for this model
                model_df = plot_df[plot_df["model_name"] == model]
                model_scores = scored_df[scored_df["model_name"] == model]
                
                # Get score for title
                if not model_scores.empty:
                    # Prioritize valid query score if available
                    valid_scores = model_scores[model_scores["null_query"] == 0]
                    if not valid_scores.empty:
                        score = valid_scores["weighted_score"].iloc[0]
                    else:
                        score = model_scores["weighted_score"].iloc[0]
                        
                    title_text = f"{model}\nScore: {score:.2f}"
                else:
                    title_text = model
                    
                ax.set_title(title_text, fontsize=10, fontweight="bold")
                
                # Plot each query type
                for _, row in model_df.iterrows():
                    null_query = row["null_query"]
                    
                    # Get values
                    values = [row[m] for m in metrics]
                    values += values[:1]  # Close loop
                    
                    # Style
                    color = "#4285F4" if null_query == 0 else "#DB4437"  # Blue for valid, Red for null
                    style = "solid" if null_query == 0 else "dashed"
                    label = "Valid" if null_query == 0 else "Null"
                    
                    # Plot
                    ax.plot(angles, values, linewidth=2, linestyle=style, color=color, label=label)
                    ax.fill(angles, values, color=color, alpha=0.1)
                
                # Add legend if multiple query types
                if len(model_df) > 1:
                    ax.legend(loc="upper right", fontsize=8)
            
            # Add overall title
            title = f"{self.test.replace('_', '-').title()} Model Performance"
            if null_filter == 0:
                title += " - Valid Queries"
            elif null_filter == 1:
                title += " - Null Queries"
                
            plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
            plt.tight_layout()
            
        else:
            # Create a single chart with all models
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
            
            # Set axes
            ax.set_theta_offset(pi/2)  # Start at top
            ax.set_theta_direction(-1)  # Go clockwise
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1)
            
            # Style
            ax.grid(alpha=0.3)
            ax.set_facecolor("#f8f9fa")
            
            # Colors and styles
            colors = sns.color_palette("viridis", len(models))
            color_map = {model: colors[i] for i, model in enumerate(models)}
            styles = {0: "solid", 1: "dashed"}
            
            # Plot each model
            for _, row in scored_df.iterrows():
                model = row["model_name"]
                null_query = row["null_query"]
                
                # Get score for label
                score = row["weighted_score"]
                score_text = f" (Score: {score:.2f})"
                
                # Get values
                values = [row[m] for m in metrics]
                values += values[:1]  # Close loop
                
                # Style
                color = color_map[model]
                style = styles.get(null_query, "solid")
                
                # Label with score
                if null_filter is not None:
                    label = f"{model}{score_text}"
                else:
                    label = f"{model} ({'Null' if null_query else 'Valid'}){score_text}"
                
                # Plot
                ax.plot(angles, values, linewidth=2.5, linestyle=style, color=color, label=label)
                ax.fill(angles, values, color=color, alpha=0.1)
            
            # Title and legend
            title = f"{self.test.replace('_', '-').title()} Model Performance"
            if null_filter == 0:
                title += " - Valid Queries"
            elif null_filter == 1:
                title += " - Null Queries"
                
            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        
        # Save if output file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        
        return fig

    def bar_chart(self, df, metric="results_match", output_file=None):
        """Create bar chart comparing models by metric"""
        sns.set_style("whitegrid")
        models = df["model_name"].unique()
        
        # Prepare data for bar chart
        data = []
        for model in models:
            valid = df[(df["model_name"] == model) & (df["null_query"] == 0)][metric].values
            null = df[(df["model_name"] == model) & (df["null_query"] == 1)][metric].values
            all_rows = df[(df["model_name"] == model) & (df["null_query"] == "overall")][metric].values
            
            data.append({
                "model": model,
                "Valid": valid[0] if len(valid) > 0 else 0,
                "Null": null[0] if len(null) > 0 else 0,
                "Overall": all_rows[0] if len(all_rows) > 0 else 0,
            })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Bar settings
        width = 0.25
        x = np.arange(len(models))
        colors = ["#4285F4", "#DB4437", "#0F9D58"]  # Blue, Red, Green
        
        # Plot each category (Valid, Null, Overall)
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
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + width / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )
        
        # Format chart
        metric_name = " ".join(word.capitalize() for word in metric.split("_"))
        ax.set_title(
            f"{self.test.replace('_', '-').title()} {metric_name} by Model ({self.model_type.capitalize()})",
        )
        ax.set_ylim(0, 1)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        
        plt.tight_layout()
        
        # Save if output file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        
        return fig, ax

    def generate_all_charts(self, output_prefix="./"):
        """Generate all charts and save them to files"""
        if self.scores is None:
            raise ValueError("Scores not calculated. Run calculate_scores() first.")
            
        # Maps for output file naming
        output_map = {1: "null", 0: "not_null", "overall": "all"}
        
        # Generate radar plots for different null_query filters
        for i in [0, 1, "overall"]:
            self.radar_plot(
                self.scores,
                null_filter=i,
                group_chart=True,
                output_file=f"{output_prefix}{output_map[i]}_grouped_radar.png",
            )
        
        # Generate bar charts for each metric
        for metric in [
            "table_access_accuracy",
            "column_access_accuracy",
            "query_similarity",
            "results_match",
        ]:
            self.bar_chart(
                self.scores, 
                metric=metric, 
                output_file=f"{output_prefix}{metric}_bar.png"
            )
            
        # Create weighted score bar chart
        # First, add weighted scores to the scores dataframe
        scored_df = self.calculate_score(self.scores)
        self.bar_chart(
            scored_df,
            metric="weighted_score",
            output_file=f"{output_prefix}weighted_score_bar.png"
        )
        
        print(f"All charts generated and saved to {output_prefix}")

    def save_analysis(self, output_path=None):
        """Save analysis dataframe and scores to CSV files"""
        if self.analysis_df is None:
            raise ValueError("Analysis dataframe not found. Run process_results() first.")
            
        if output_path is None:
            output_path = "./"
            
        # Save analysis dataframe
        analysis_file = f"{output_path}{self.test}_analysis.csv"
        self.analysis_df.to_csv(analysis_file, index=False)
        print(f"Analysis saved to {analysis_file}")
        
        # Save scores if calculated
        if self.scores is not None:
            scores_file = f"{output_path}{self.test}_scores.csv"
            self.scores.to_csv(scores_file, index=False)
            print(f"Scores saved to {scores_file}")