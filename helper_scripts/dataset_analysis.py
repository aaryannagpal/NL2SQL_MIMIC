import pandas as pd
import numpy as np
import re, sys, os
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import TRAINING_DATA, STORE_DATASET_ANALYSIS

df = pd.read_csv(TRAINING_DATA)
# Basic info
print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())

# Display a few sample rows
df.head(3)

na_queries = df['true_query'].isna().sum()
non_na_queries = len(df) - na_queries
print(f"Number of NaN true_query values: {na_queries} ({na_queries/len(df)*100:.2f}%)")
print(f"Number of non-NaN true_query values: {non_na_queries} ({non_na_queries/len(df)*100:.2f}%)")

labels = ['With valid NL Query', 'Without valid NL Query']
values = [non_na_queries, na_queries]
colors = ['royalblue', 'lightcoral']

fig_pie = px.pie(
    values=values,
    names=labels,
    # title='Distribution of Questions with and without SQL Queries',
    color_discrete_sequence=colors
)
fig_pie.update_traces(textinfo='percent+label')
fig_pie.update_layout(showlegend=False)
fig_pie.write_image(os.path.join(STORE_DATASET_ANALYSIS, "query_distribution_pie_chart.png"), scale=2)


def count_words(text):
    return len(str(text).split())

question_starters = ['what', 'how', 'when', 'where', 'which', 'who', 'why', 'is', 'are', 'can', 'do', 'does']


def get_question_type(question):
    if not isinstance(question, str):
        return 'other'
    
    first_word = question.lower().split()[0]
    if first_word in question_starters:
        return first_word
    return 'other'

df['question_length'] = df['question'].apply(len)
df['question_word_count'] = df['question'].apply(count_words)
df['question_type'] = df['question'].apply(get_question_type)
df['question_char_count'] = df['question'].apply(len)

# Create separate dataframes for questions with and without SQL
df_with_sql = df[~df['true_query'].isna()]
df_without_sql = df[df['true_query'].isna()]

# Create a histogram showing the distribution of question lengths
fig_hist = make_subplots(rows=1, cols=2, subplot_titles=('Valid NL Questions', 'Invalid NL Questions'))

fig_hist.add_trace(
    go.Histogram(
        x=df_with_sql['question_length'],
        name='Valid NL Queries',
        marker_color='royalblue',
        opacity=0.7
    ),
    row=1, col=1
)

fig_hist.add_trace(
    go.Histogram(
        x=df_without_sql['question_length'],
        name='Invalid NL Queries',
        marker_color='lightcoral',
        opacity=0.7
    ),
    row=1, col=2
)

fig_hist.update_layout(
    # title_text='Distribution of Length of Questions',
    xaxis_title='Character Length',
    yaxis_title='Count',
    barmode='overlay',
    height=600,
    width=1000,
    showlegend=False
)

fig_hist.write_image(os.path.join(STORE_DATASET_ANALYSIS, 'q_length_dist.png'), scale = 2)



# Create a box plot comparing word counts for questions with and without SQL
fig_box = go.Figure()

fig_box.add_trace(go.Box(
    y=df_with_sql['question_word_count'],
    name='With SQL',
    marker_color='royalblue',
    boxmean=True
))

fig_box.add_trace(go.Box(
    y=df_without_sql['question_word_count'],
    name='Without SQL',
    marker_color='lightcoral',
    boxmean=True
))

fig_box.update_layout(
    # title_text='Question Word Count Comparison',
    yaxis_title='Word Count',
    height=600,
    width=1000,
    showlegend=False
)

fig_box.write_image(os.path.join(STORE_DATASET_ANALYSIS, 'q_word_count_comp.png'), scale = 2)




from collections import Counter
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def get_word_freq(text_series):
    words = []
    for text in text_series:
        if isinstance(text, str):
            # Remove punctuation and convert to lowercase
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            # Split into words and exclude stopwords
            words.extend([word for word in clean_text.split() if word not in stop_words])
    return Counter(words)

# Get word frequencies for questions with and without SQL
with_sql_words = get_word_freq(df_with_sql['question'])
without_sql_words = get_word_freq(df_without_sql['question'])

# Convert to DataFrames for visualization
with_sql_df = pd.DataFrame(with_sql_words.most_common(15), columns=['word', 'count'])
without_sql_df = pd.DataFrame(without_sql_words.most_common(15), columns=['word', 'count'])

fig_words = make_subplots(rows=1, cols=2, subplot_titles=('Common Words in Questions with SQL', 
                                                          'Common Words in Questions without SQL'))

fig_words.add_trace(
    go.Bar(
        x=with_sql_df['word'],
        y=with_sql_df['count'],
        marker_color='royalblue'
    ),
    row=1, col=1
)

fig_words.add_trace(
    go.Bar(
        x=without_sql_df['word'],
        y=without_sql_df['count'],
        marker_color='lightcoral'
    ),
    row=1, col=2
)

fig_words.update_layout(
    # title_text='Most Common Words in Questions',
    xaxis_title='Word',
    yaxis_title='Frequency',
    height=600,
    width=900,
    showlegend=False
)

fig_words.write_image(os.path.join(STORE_DATASET_ANALYSIS, 'most_common_words.png'), scale = 2)


# Calculate percentages for questions with SQL
with_sql_counts = df_with_sql['question_type'].value_counts()
with_sql_total = with_sql_counts.sum()
with_sql_pct = (with_sql_counts / with_sql_total * 100).reset_index()
with_sql_pct.columns = ['type', 'percentage']
with_sql_pct['has_sql'] = 'With SQL'

# Calculate percentages for questions without SQL
without_sql_counts = df_without_sql['question_type'].value_counts()
without_sql_total = without_sql_counts.sum()
without_sql_pct = (without_sql_counts / without_sql_total * 100).reset_index()
without_sql_pct.columns = ['type', 'percentage']
without_sql_pct['has_sql'] = 'Without SQL'

# Combine the data
question_types_pct = pd.concat([with_sql_pct, without_sql_pct])

# Part 2: Analyzing SQL Query Structure and Complexity
# This builds on the existing analysis you've already done

# First, let's create functions to extract SQL components and measure complexity
import re

def count_sql_components(query):
    """Count various SQL components in a query"""
    if not isinstance(query, str):
        return {
            'select_count': 0,
            'from_count': 0,
            'where_count': 0,
            'join_count': 0,
            'group_by_count': 0,
            'order_by_count': 0,
            'limit_count': 0,
            'distinct_count': 0,
            'subquery_count': 0,
            'table_count': 0,
            'column_count': 0
        }
    
    query = query.lower()
    
    # Count basic SQL clauses
    components = {
        'select_count': query.count('select'),
        'from_count': query.count('from'),
        'where_count': query.count('where'),
        'join_count': sum(query.count(j) for j in [' join ', ' inner join ', ' left join ', ' right join ']),
        'group_by_count': query.count('group by'),
        'order_by_count': query.count('order by'),
        'limit_count': query.count('limit'),
        'distinct_count': query.count('distinct'),
    }
    
    # Count subqueries (simplified approach)
    components['subquery_count'] = query.count('(select')
    
    # Count tables and columns (simplified approach)
    tables = set(re.findall(r'from\s+([a-zA-Z0-9_]+)', query))
    tables.update(re.findall(r'join\s+([a-zA-Z0-9_]+)', query))
    components['table_count'] = len(tables)
    
    # Extract columns from SELECT clause
    select_pattern = r'select\s+(.*?)\s+from'
    select_match = re.search(select_pattern, query)
    columns = set()
    if select_match:
        select_clause = select_match.group(1)
        if '*' not in select_clause:
            # Split by commas, but be careful of functions like COUNT(col)
            # This is a simplified approach
            cols = re.findall(r'([a-zA-Z0-9_\.]+)(?:\s*,|\s*$|\s+as)', select_clause)
            columns.update(cols)
    
    # Extract columns from WHERE clause
    where_columns = re.findall(r'where\s+([a-zA-Z0-9_\.]+)\s*[=<>]', query)
    columns.update(where_columns)
    
    components['column_count'] = len(columns)
    
    return components

def calculate_query_complexity(components):
    """Calculate overall query complexity based on components"""
    # Simple weighted sum of components
    weights = {
        'select_count': 1,
        'from_count': 1,
        'where_count': 1.5,
        'join_count': 2,
        'group_by_count': 1.5,
        'order_by_count': 1,
        'limit_count': 0.5,
        'distinct_count': 1,
        'subquery_count': 3,
        'table_count': 1.5,
        'column_count': 0.5
    }
    
    complexity = sum(components[key] * weights[key] for key in weights)
    return complexity


# Apply these functions to the non-NaN queries
# First ensure we're only working with the valid queries
df_with_sql = df[~df['true_query'].isna()].copy()

# Extract SQL components
sql_components = df_with_sql['true_query'].apply(count_sql_components)
df_with_sql = pd.concat([df_with_sql, pd.json_normalize(sql_components)], axis=1)

# Calculate overall complexity
for component in ['select_count', 'from_count', 'where_count', 'join_count', 
                 'group_by_count', 'order_by_count', 'limit_count', 'distinct_count', 
                 'subquery_count', 'table_count', 'column_count']:
    if component not in df_with_sql.columns:
        df_with_sql[component] = 0

df_with_sql['query_complexity'] = df_with_sql.apply(
    lambda row: calculate_query_complexity({
        'select_count': row['select_count'],
        'from_count': row['from_count'],
        'where_count': row['where_count'],
        'join_count': row['join_count'],
        'group_by_count': row['group_by_count'],
        'order_by_count': row['order_by_count'],
        'limit_count': row['limit_count'],
        'distinct_count': row['distinct_count'],
        'subquery_count': row['subquery_count'],
        'table_count': row['table_count'],
        'column_count': row['column_count']
    }), 
    axis=1
)

# Map complexity scores to complexity levels
def map_complexity_level(score):
    if score < 5:
        return "Simple"
    elif score < 10:
        return "Moderate"
    elif score < 15:
        return "Complex"
    else:
        return "Very Complex"

df_with_sql['complexity_level'] = df_with_sql['query_complexity'].apply(map_complexity_level)

# Basic statistics of query complexity
print("\nSQL Query Complexity Statistics:")
print(df_with_sql['query_complexity'].describe())

# Visualize distribution of complexity levels
complexity_counts = df_with_sql['complexity_level'].value_counts().reset_index()
complexity_counts.columns = ['complexity_level', 'count']

# Create ordered categories
complexity_order = ['Simple', 'Moderate', 'Complex', 'Very Complex']
complexity_counts['complexity_level'] = pd.Categorical(
    complexity_counts['complexity_level'], 
    categories=complexity_order, 
    ordered=True
)
complexity_counts = complexity_counts.sort_values('complexity_level')

fig_complexity_levels = px.bar(
    complexity_counts,
    x='complexity_level',
    y='count',
    # title='Distribution of SQL Query Complexity Levels',
    color='complexity_level',
    color_discrete_map={
        'Simple': 'lightblue',
        'Moderate': 'royalblue',
        'Complex': 'darkblue',
        'Very Complex': 'purple'
    }
)

fig_complexity_levels.update_layout(
    xaxis_title='Complexity Level',
    yaxis_title='Count',
    height=500,
    width=900,
    showlegend=False
)

fig_complexity_levels.write_image(os.path.join(STORE_DATASET_ANALYSIS, 'complexity_level_dist.png'), scale = 2)


# Also show the original complexity score distribution
fig_complexity = px.histogram(
    df_with_sql, 
    x='query_complexity',
    # title='Distribution of SQL Query Complexity Scores',
    labels={'query_complexity': 'Complexity Score'},
    color='complexity_level',
    color_discrete_map={
        'Simple': 'lightblue',
        'Moderate': 'royalblue',
        'Complex': 'darkblue',
        'Very Complex': 'purple'
    }
)

fig_complexity.update_layout(
    xaxis_title='Complexity Score',
    yaxis_title='Count',
    height=500,
    width=900
)

fig_complexity.write_image(os.path.join(STORE_DATASET_ANALYSIS, 'complexity_score_dist.png'), scale = 2)

component_cols = ['select_count', 'from_count', 'where_count', 'join_count', 
                  'group_by_count', 'order_by_count', 'limit_count', 'distinct_count', 
                  'subquery_count']

# Calculate averages for each component
component_avgs = df_with_sql[component_cols].mean().reset_index()
component_avgs.columns = ['component', 'average']

def identify_question_intent(question):
    """
    Identify the primary intent of a question based on keywords and structure
    """
    if not isinstance(question, str):
        return "Unknown"
    
    question = question.lower().strip()
    
    # Different types of question intents
    intents = {
        'Count/Quantity': ['how many', 'count', 'number of', 'total number', 'total amount'],
        'Existence': ['is there', 'are there', 'has', 'have', 'did', 'does', 'do'],
        'Time/When': ['when', 'what time', 'what date', 'how long', 'how often'],
        'Value/Measure': ['how much', 'what is the value', 'what was the', 'average', 'mean', 'median', 'maximum', 'minimum'],
        'Comparison': ['more than', 'less than', 'greater than', 'higher than', 'lower than', 'compared to', 'versus'],
        'List/Enumeration': ['what are', 'list', 'show me', 'display', 'give me all'],
        'Latest/Recent': ['latest', 'most recent', 'last', 'newest'],
        'First/Earliest': ['first', 'earliest', 'oldest', 'initial']
    }
    
    # Check for each intent type
    for intent, phrases in intents.items():
        if any(phrase in question for phrase in phrases):
            return intent
    
    # Default intents based on first word
    first_word = question.split()[0] if question else ""
    
    if first_word == 'what':
        return 'Information'
    elif first_word == 'who':
        return 'Person'
    elif first_word == 'where':
        return 'Location'
    elif first_word == 'why':
        return 'Reason'
    elif first_word == 'how':
        return 'Method/Process'
    
    return 'General Query'

# Apply intent identification to all questions
df['question_intent'] = df['question'].apply(identify_question_intent)

# Visualize question intent distribution
intent_counts = df['question_intent'].value_counts().reset_index()
intent_counts.columns = ['intent', 'count']

fig_intent = px.bar(
    intent_counts,
    x='intent',
    y='count',
    # title='Question Intent Distribution',
    # color='count',
    # color_continuous_scale='Viridis',
    labels={
        'intent': 'Question Intent',
        'count': 'Count'
    }
)

fig_intent.update_layout(
    xaxis_title='Intent',
    yaxis_title='Count',
    height=500,
    width=900,
    xaxis={'categoryorder':'total descending'}
)

fig_intent.write_image(os.path.join(STORE_DATASET_ANALYSIS, 'question_intent_dist.png'), scale = 2)

def analyze_result_structure(result):
    """
    Analyze the structure of a single query result
    Returns a dictionary with structural information
    """
    if not isinstance(result, list) or len(result) == 0:
        return {
            'row_count': 0,
            'column_count': 0,
            'is_empty': True,
            'has_aggregation': False,
            'result_type': 'empty'
        }
    
    # Extract information about the result structure
    row_count = len(result)
    
    # Check if result contains dictionaries (standard format)
    if isinstance(result[0], dict):
        column_count = len(result[0].keys())
        column_names = list(result[0].keys())
        
        # Check for aggregation functions in column names
        agg_functions = ['count', 'sum', 'avg', 'min', 'max', 'distinct']
        has_aggregation = any(
            any(agg.lower() in col.lower() for agg in agg_functions)
            for col in column_names
        )
        
        # Determine result type
        if row_count == 1 and column_count == 1:
            result_type = 'single_value'
        elif row_count == 1:
            result_type = 'single_row'
        elif column_count == 1:
            result_type = 'single_column'
        else:
            result_type = 'table'
    else:
        # Handle non-standard results
        column_count = 1
        has_aggregation = False
        result_type = 'non_standard'
    
    return {
        'row_count': row_count,
        'column_count': column_count,
        'is_empty': False,
        'has_aggregation': has_aggregation,
        'result_type': result_type
    }

# Function to safely parse result JSON
def parse_result(result_str):
    """Parse JSON result string or return empty list if invalid"""
    if pd.isna(result_str):
        return []
    
    if isinstance(result_str, list):
        return result_str
    
    if isinstance(result_str, str):
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            return []
    
    return []

# Apply to dataframe (assuming we have a df_with_sql dataframe with 'results' column)
# If 'results' column doesn't exist yet, we need to create it
if 'results' not in df_with_sql.columns:
    # Check if there might be a different column name for results
    possible_result_columns = ['result', 'results', 'true_result']
    for col in possible_result_columns:
        if col in df_with_sql.columns:
            df_with_sql['results'] = df_with_sql[col].apply(parse_result)
            break
    
    # If still no results column, create empty one
    if 'results' not in df_with_sql.columns:
        df_with_sql['results'] = [[]] * len(df_with_sql)

# Now analyze result structures
result_structures = df_with_sql['results'].apply(analyze_result_structure)
df_with_sql = pd.concat([df_with_sql, pd.json_normalize(result_structures)], axis=1)

# Visualize distribution of result types
result_type_counts = df_with_sql['result_type'].value_counts().reset_index()
result_type_counts.columns = ['result_type', 'count']

fig_result_types = px.bar(
    result_type_counts,
    x='result_type',
    y='count',
    # title='Distribution of SQL Query Result Types',
    labels={
        'result_type': 'Result Type',
        'count': 'Count'
    }
)

fig_result_types.update_layout(
    xaxis_title='Result Type',
    yaxis_title='Count',
    height=500,
    width=900
)

fig_result_types.write_image(os.path.join(STORE_DATASET_ANALYSIS, 'empty_res_dist.png'), scale = 2)