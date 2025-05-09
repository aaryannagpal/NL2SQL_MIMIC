import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import plotly.io as pio

pio.templates.default = "plotly_white"

COLORS = px.colors.qualitative.D3

def process_stages_comparison(
    stage_results: Dict[str, pd.DataFrame],
    output_dir: str,
    stage_order: List[str] = ["zeroshot", "fewshot", "schema_aware"]
):
    """
    Generate visualizations comparing model performance across different stages.
    
    Args:
        stage_results: Dictionary with stage names as keys and dataframes as values
        output_dir: Directory to save visualizations
        stage_order: Order of stages for x-axis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    combined_df = pd.DataFrame()
    for stage, df in stage_results.items():
        stage_df = df.copy()
        stage_df['experiment'] = stage  # Ensure experiment column is set correctly
        combined_df = pd.concat([combined_df, stage_df], ignore_index=True)
    
    metrics = [
        'execution_success_rate', 
        'result_match_rate', 
        'avg_table_access_accuracy', 
        'avg_column_access_accuracy', 
        'avg_query_similarity',
        'avg_select_similarity', 
        'avg_from_similarity', 
        'avg_where_similarity', 
        'avg_composite_score'
    ]
    
    metric_display_names = {
        'execution_success_rate': 'Execution Success Rate', 
        'result_match_rate': 'Result Match Rate', 
        'avg_table_access_accuracy': 'Table Access Accuracy', 
        'avg_column_access_accuracy': 'Column Access Accuracy', 
        'avg_query_similarity': 'Query Similarity',
        'avg_select_similarity': 'SELECT Clause Similarity', 
        'avg_from_similarity': 'FROM Clause Similarity', 
        'avg_where_similarity': 'WHERE Clause Similarity', 
        'avg_composite_score': 'Composite Score'
    }
    
    stage_display_names = {
        'zeroshot': 'Zero-Shot',
        'fewshot': 'Few-Shot',
        'schema_aware': 'Schema-Aware',
        'finetune': 'Fine-Tuning\n(Evaluated on Test Data)',
    }
    
    models = combined_df['model'].unique()
    
    for metric in metrics:
        if metric not in combined_df.columns:
            print(f"Metric {metric} not found in dataframe")
            continue
            
        fig = go.Figure()
        
        for i, model in enumerate(models):
            model_data = combined_df[combined_df['model'] == model]
            
            ordered_data = []
            for stage in stage_order:
                stage_data = model_data[model_data['experiment'] == stage]
                if not stage_data.empty:
                    ordered_data.append(stage_data)
            
            if ordered_data:
                ordered_df = pd.concat(ordered_data, ignore_index=True)
                
                ordered_df['display_stage'] = ordered_df['experiment'].map(stage_display_names)
                
                fig.add_trace(go.Scatter(
                    x=ordered_df['display_stage'],
                    y=ordered_df[metric],
                    mode='lines+markers',
                    name=model,
                    line=dict(width=2.5, color=COLORS[i % len(COLORS)]),
                    marker=dict(size=10, line=dict(width=1, color='white'))
                ))
        
        fig.update_layout(
            title=dict(
                text=metric_display_names.get(metric, metric),
                font=dict(size=18, family="Arial", color="#333"),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title="Prompting Strategy",
                tickfont=dict(size=12),
                categoryorder='array',
                categoryarray=[stage_display_names.get(s, s) for s in stage_order],
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title=metric_display_names.get(metric, metric),
                tickfont=dict(size=12),
                range=[0, 1] if 'rate' in metric or 'accuracy' in metric or 'similarity' in metric or 'score' in metric else None,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.1)',
                tickformat=".2f"
            ),
            legend=dict(
                font=dict(size=12, family="Arial"),
                borderwidth=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)'
            ),
            width=900,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        metric_name = metric.replace('avg_', '').replace('_', '_')
        fig.write_image(os.path.join(output_dir, f"{metric_name}_by_model.png"), scale=2)
        
    fig = go.Figure()
    
    stage_averages = {}
    for stage in stage_order:
        stage_df = combined_df[combined_df['experiment'] == stage]
        if not stage_df.empty:
            stage_averages[stage] = {metric: stage_df[metric].mean() for metric in metrics if metric in stage_df}
    
    for i, metric in enumerate(metrics):
        if not all(metric in stage_averages[stage] for stage in stage_averages):
            continue
        
        display_stages = [stage_display_names.get(stage, stage) for stage in stage_averages.keys()]
            
        fig.add_trace(go.Scatter(
            x=display_stages,
            y=[stage_averages[stage][metric] for stage in stage_averages],
            mode='lines+markers',
            name=metric_display_names.get(metric, metric),
            line=dict(width=2.5, color=COLORS[i % len(COLORS)]),
            marker=dict(size=10, line=dict(width=1, color='white'))
        ))
    
    fig.update_layout(
        title=dict(
            text="Performance Metrics Across Prompting Strategies",
            font=dict(size=18, family="Arial", color="#333"),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title="Prompting Strategy",
            tickfont=dict(size=12),
            categoryorder='array',
            categoryarray=[stage_display_names.get(s, s) for s in stage_order],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title="Performance Value",
            tickfont=dict(size=12),
            range=[0, 1],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(0,0,0,0.1)',
            tickformat=".2f"
        ),
        legend=dict(
            font=dict(size=10, family="Arial"),
            orientation="v",
            xanchor="left",
            x=1.02,
            yanchor="top",
            y=1,
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)'
        ),
        width=900,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=80, r=200, t=100, b=80)  # Extra right margin for legend
    )
    
    fig.write_image(os.path.join(output_dir, "metrics_average_by_stage.png"), scale=3, width=1200, height=800)
    
    print(f"All visualizations saved to {output_dir}")
    
    return combined_df


if __name__ == "__main__":
    zeroshot_results = pd.read_csv("/home/aaryan/Documents/Ashoka/Sem_8/Capstone_Thesis/NL2SQL_MIMIC/analysis/m2/zeroshot/zeroshot.csv")
    fewshot_results = pd.read_csv("/home/aaryan/Documents/Ashoka/Sem_8/Capstone_Thesis/NL2SQL_MIMIC/analysis/m2/fewshot/fewshot.csv")
    schema_aware_results = pd.read_csv("/home/aaryan/Documents/Ashoka/Sem_8/Capstone_Thesis/NL2SQL_MIMIC/analysis/m2/schema_aware/schema_aware.csv")
    finetune_results = pd.read_csv("/home/aaryan/Documents/Ashoka/Sem_8/Capstone_Thesis/NL2SQL_MIMIC/analysis/m2/finetune/finetune.csv")
    
    stage_results = {
        "zeroshot": zeroshot_results,
        "fewshot": fewshot_results,
        "schema_aware": schema_aware_results,
        "finetune": finetune_results
    }
    
    process_stages_comparison(
        stage_results=stage_results,
        output_dir="visualizations/comparison",
        stage_order=["zeroshot", "fewshot", "schema_aware", "finetune"]
    )