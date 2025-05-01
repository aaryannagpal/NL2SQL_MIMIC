import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os, sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.query_analyzer import QueryAnalyzer

def analyze_model_results(
    ground_truth_path: str,
    model_results_paths: Dict[str, str],
    output_dir: str,
    to_compare: str = 'generated_sql'
) -> None:
    """
    Analyze and visualize query performance across multiple models.
    
    Args:
        ground_truth_path: Path to CSV with ground truth data
        model_results_paths: Dictionary of model_name -> results_csv_path
        output_dir: Directory to save visualizations
        to_compare: Column name with generated queries
        
    Returns:
        None (saves visualizations to output_dir)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth data
    ground_truth_df = pd.read_csv(ground_truth_path)
    print(f"Loaded ground truth data with {len(ground_truth_df)} queries")
    
    # Initialize QueryAnalyzer
    analyzer = QueryAnalyzer()
    
    # Analyze each model and collect results
    all_results = {}
    aggregate_metrics = []
    
    for model_name, results_path in model_results_paths.items():
        print(f"Analyzing {model_name}...")
        model_df = pd.read_csv(results_path)
        
        # Run analysis
        analysis_df = analyzer.compare_queries(
            model_df, 
            true_df=ground_truth_df, 
            to_compare=to_compare
        )
        
        # Store results
        all_results[model_name] = analysis_df
        
        # Collect aggregate metrics
        metrics = {
            "model": model_name,
            "total_queries": len(analysis_df),
            "execution_success_rate": analysis_df["gen_success"].mean() if "gen_success" in analysis_df.columns else np.nan,
            "result_match_rate": analysis_df["results_match"].mean() if "results_match" in analysis_df.columns else np.nan,
            "avg_table_access_accuracy": analysis_df["table_access_accuracy"].mean(),
            "avg_column_access_accuracy": analysis_df["column_access_accuracy"].mean(),
            "avg_query_similarity": analysis_df["query_similarity"].mean(),
            "avg_select_similarity": analysis_df["select_clause_similarity"].mean() if "select_clause_similarity" in analysis_df.columns else np.nan,
            "avg_from_similarity": analysis_df["from_clause_similarity"].mean() if "from_clause_similarity" in analysis_df.columns else np.nan,
            "avg_where_similarity": analysis_df["where_clause_similarity"].mean() if "where_clause_similarity" in analysis_df.columns else np.nan,
            "avg_composite_score": analysis_df["composite_score"].mean() if "composite_score" in analysis_df.columns else np.nan,
            "empty_results_both": analysis_df["both_empty"].sum() if "both_empty" in analysis_df.columns else np.nan,
        }
        aggregate_metrics.append(metrics)
    
    # Create aggregate metrics dataframe
    metrics_df = pd.DataFrame(aggregate_metrics)
    
    # Generate visualizations
    generate_overall_performance_chart(metrics_df, output_dir)
    generate_component_similarity_chart(metrics_df, output_dir)
    generate_structural_accuracy_chart(metrics_df, output_dir)
    
    # Generate error analysis charts for each model
    for model_name, analysis_df in all_results.items():
        if "gen_error_category" in analysis_df.columns:
            generate_error_category_chart(analysis_df, model_name, output_dir)
    
    if any("gen_error_category" in df.columns for df in all_results.values()):
        generate_combined_error_categories(all_results, output_dir)
    # Generate query complexity analysis if possible
    if all("query_complexity" in df.columns for df in all_results.values()):
        generate_complexity_performance_chart(all_results, output_dir)

    # Generate comparison chart for empty results handling
    if all("both_empty" in df.columns for df in all_results.values()):
        generate_empty_results_chart(all_results, output_dir)
    
    print(f"All visualizations saved to {output_dir}")


def generate_overall_performance_chart(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """Generate chart showing overall performance metrics by model"""
    # Select relevant metrics for overall performance
    performance_metrics = [
        "execution_success_rate", 
        "result_match_rate", 
        "avg_composite_score"
    ]
    
    # Filter and reshape data for plotting
    plot_data = metrics_df.melt(
        id_vars=["model"], 
        value_vars=performance_metrics,
        var_name="Metric", 
        value_name="Value"
    )
    
    # Create visualization
    fig = px.bar(
        plot_data, 
        x="model", 
        y="Value", 
        color="Metric", 
        barmode="group",
        category_orders={"Metric": performance_metrics}
    )
    
    # Update layout
    fig.update_layout(
        yaxis_title="Score",
        xaxis_title="",
        legend_title_text="",
        yaxis=dict(range=[0, 1])  # Force y-axis to be 0-1 for percentages
    )
    
    # Save visualization
    fig.write_image(os.path.join(output_dir, "overall_performance.png"))

def generate_component_similarity_chart(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """Generate chart showing component-level similarity by model"""
    # Select component similarity metrics
    component_metrics = [
        "avg_select_similarity", 
        "avg_from_similarity", 
        "avg_where_similarity"
    ]
    
    # Only include metrics that exist in the data
    available_metrics = [m for m in component_metrics if m in metrics_df.columns]
    
    if not available_metrics:
        print("No component similarity metrics available")
        return
    
    # Filter and reshape data for plotting
    plot_data = metrics_df.melt(
        id_vars=["model"], 
        value_vars=available_metrics,
        var_name="Component", 
        value_name="Similarity"
    )
    
    # Clean up component names for display
    plot_data["Component"] = plot_data["Component"].str.replace("avg_", "").str.replace("_similarity", "")
    
    # Create visualization
    fig = px.bar(
        plot_data, 
        x="model", 
        y="Similarity", 
        color="Component", 
        barmode="group"
    )
    
    # Update layout
    fig.update_layout(
        yaxis_title="Similarity",
        xaxis_title="",
        legend_title_text="",
        yaxis=dict(range=[0, 1])
    )
    
    # Save visualization
    fig.write_image(os.path.join(output_dir, "component_similarity.png"))

def generate_structural_accuracy_chart(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """Generate chart showing structural accuracy metrics by model"""
    # Select structural accuracy metrics
    structural_metrics = [
        "avg_table_access_accuracy", 
        "avg_column_access_accuracy", 
        "avg_query_similarity"
    ]
    
    # Filter and reshape data for plotting
    plot_data = metrics_df.melt(
        id_vars=["model"], 
        value_vars=structural_metrics,
        var_name="Metric", 
        value_name="Value"
    )
    
    # Clean up metric names for display
    plot_data["Metric"] = plot_data["Metric"].str.replace("avg_", "").str.replace("_", " ")
    
    # Create visualization
    fig = px.bar(
        plot_data, 
        x="model", 
        y="Value", 
        color="Metric", 
        barmode="group"
    )
    
    # Update layout
    fig.update_layout(
        yaxis_title="Accuracy",
        xaxis_title="",
        legend_title_text="",
        yaxis=dict(range=[0, 1])
    )
    
    # Save visualization
    fig.write_image(os.path.join(output_dir, "structural_accuracy.png"))

def generate_error_category_chart(analysis_df: pd.DataFrame, model_name: str, output_dir: str) -> None:
    """Generate chart showing error categories for a model"""
    # Count error categories
    if "gen_error_category" not in analysis_df.columns:
        return
        
    error_counts = analysis_df["gen_error_category"].value_counts().reset_index()
    error_counts.columns = ["Category", "Count"]
    
    # Create visualization
    fig = px.pie(
        error_counts, 
        values="Count", 
        names="Category", 
        hole=0.4
    )
    
    # Update layout
    fig.update_layout(
        legend_title_text=""
    )
    
    # Save visualization
    fig.write_image(os.path.join(output_dir, f"{model_name}_error_categories.png"))

def generate_complexity_performance_chart(all_results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Generate chart showing performance by query complexity across models"""
    # Prepare data for plotting
    plot_data = []
    
    for model_name, df in all_results.items():
        if "query_complexity" not in df.columns or "results_match" not in df.columns:
            continue
            
        # Bin query complexity
        df['complexity_bin'] = pd.cut(
            df['query_complexity'], 
            bins=[0, 5, 10, 15, float('inf')], 
            labels=['Simple', 'Moderate', 'Complex', 'Very Complex']
        )
        
        # Calculate success rate by complexity bin
        complexity_perf = df.groupby('complexity_bin')['results_match'].mean().reset_index()
        complexity_perf['model'] = model_name
        
        plot_data.append(complexity_perf)
    
    if not plot_data:
        print("No complexity performance data available")
        return
        
    # Combine data from all models
    combined_data = pd.concat(plot_data)
    
    # Create visualization
    fig = px.line(
        combined_data, 
        x="complexity_bin", 
        y="results_match", 
        color="model", 
        markers=True,
        category_orders={"complexity_bin": ['Simple', 'Moderate', 'Complex', 'Very Complex']}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Query Complexity",
        yaxis_title="Result Match Rate",
        legend_title_text="",
        yaxis=dict(range=[0, 1])
    )
    
    # Save visualization
    fig.write_image(os.path.join(output_dir, "complexity_performance.png"))

def generate_empty_results_chart(all_results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Generate chart comparing handling of empty results across models"""
    # Prepare data for plotting
    empty_results_data = []
    
    for model_name, df in all_results.items():
        if not {"true_empty_result", "gen_empty_result", "both_empty", "results_match"}.issubset(df.columns):
            continue
            
        # Calculate metrics
        total = len(df)
        both_empty = df["both_empty"].sum()
        true_empty_only = (df["true_empty_result"] & ~df["gen_empty_result"]).sum()
        gen_empty_only = (~df["true_empty_result"] & df["gen_empty_result"]).sum()
        correct_nonempty = ((~df["true_empty_result"]) & (~df["gen_empty_result"]) & df["results_match"]).sum()
        
        # Add to data
        empty_results_data.append({
            "model": model_name,
            "Both Empty": both_empty / total,
            "True Empty Only": true_empty_only / total,
            "Generated Empty Only": gen_empty_only / total,
            "Correct Non-empty": correct_nonempty / total
        })
    
    if not empty_results_data:
        print("No empty results data available")
        return
        
    # Create dataframe
    empty_df = pd.DataFrame(empty_results_data)
    
    # Reshape for plotting
    plot_data = empty_df.melt(
        id_vars=["model"], 
        value_vars=["Both Empty", "True Empty Only", "Generated Empty Only", "Correct Non-empty"],
        var_name="Category", 
        value_name="Proportion"
    )
    
    # Create visualization
    fig = px.bar(
        plot_data, 
        x="model", 
        y="Proportion", 
        color="Category", 
        barmode="stack"
    )
    
    # Update layout
    fig.update_layout(
        yaxis_title="Proportion of Queries",
        xaxis_title="",
        legend_title_text=""
    )
    
    # Save visualization
    fig.write_image(os.path.join(output_dir, "empty_results_handling.png"))

def generate_combined_error_categories(all_results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Generate a combined chart showing error categories for all models in a single figure"""
    # Check if we have error categories
    models_with_errors = []
    for model_name, df in all_results.items():
        if "gen_error_category" in df.columns:
            models_with_errors.append(model_name)
    
    if not models_with_errors:
        print("No error category data available")
        return
    
    # Create subplot grid - determine rows and columns based on number of models
    n_models = len(models_with_errors)
    n_cols = min(3, n_models)  # Maximum 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplots - Each model gets its own pie chart
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        specs=[[{"type": "pie"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=models_with_errors
    )
    
    # Add pie charts for each model
    for i, model_name in enumerate(models_with_errors):
        df = all_results[model_name]
        error_counts = df["gen_error_category"].value_counts()
        
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        fig.add_trace(
            go.Pie(
                labels=error_counts.index,
                values=error_counts.values,
                hole=0.4,
                name=model_name,
                showlegend=(i == 0)  # Only show legend for first pie to avoid duplication
            ),
            row=row, 
            col=col
        )
    
    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=300 * n_cols,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Save visualization
    fig.write_image(os.path.join(output_dir, "combined_error_categories.png"))

# Example usage
