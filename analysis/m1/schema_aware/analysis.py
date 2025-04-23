import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.query_scorer import M1Scoring
from config import PROCESSED_RESULT_DIR, STORE_ANALYSIS_DIR

test = 'schema_aware'
model_type = 'm1'

print(f"Initializing M1Scoring for test: {test}, model type: {model_type}")
scorer = M1Scoring(
    results_path=PROCESSED_RESULT_DIR / model_type / test,
    analysis_path=STORE_ANALYSIS_DIR / model_type / test,
    test=test,
    model_type=model_type
)

# Process results
print("Processing result files...")
scorer.process_results()

# Calculate scores
print("Calculating scores...")
scorer.calculate_scores()

# Save analysis and scores to CSV
print("Saving analysis and scores...")
scorer.save_analysis()

# Generate all charts
print("Generating visualizations...")
scorer.generate_all_charts()

print("Done!")
