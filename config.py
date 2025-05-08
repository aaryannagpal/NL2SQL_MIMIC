import os
from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "mimic_data"

MYSQL_DB_PATH: Final[Path] = '/media/chs.gpu/DATA/nagpal/modified_mimic/data/mimic_iv/mimic_iv.sqlite'
MIMIC_SCHEMA_PATH: Final[Path] = DATA_DIR / "modified_mimic.json"
DICTIONARY_MAP_PATH: Final[Path] = DATA_DIR / "dictionary.json"

DATASET_PATH = PROJECT_ROOT / "data"
TRAINING_DATA = DATASET_PATH / "train.csv"

MODEL_DIR = PROJECT_ROOT / "model"
MODEL_LIST = MODEL_DIR / "model_list.csv"
MODELS_DIR = "/media/chs.gpu/DATA/nagpal/models/"

MIMIC_SAMPLE_DIR = PROJECT_ROOT / "data" / "sampled_MIMIC_values"
MIMIC_SAMPLE_PATH = MIMIC_SAMPLE_DIR / "default.json"

STORE_RESULT_DIR = PROJECT_ROOT / "results"
RAW_RESULT_DIR = STORE_RESULT_DIR / "raw"
PROCESSED_RESULT_DIR = STORE_RESULT_DIR / "processed"

STORE_ANALYSIS_DIR = PROJECT_ROOT / "analysis"

SAMPLE_M1_MODEL_DIR = PROJECT_ROOT / "model" / "m1" / "finetune" / "output"
SAMPLE_M2_MODEL_DIR = PROJECT_ROOT / "model" / "m2" / "finetune" / "output"

def _validate_paths() -> None:
    """Check if critical files/dirs exist at startup."""
    required_paths = [
        DATA_DIR,
        MIMIC_SCHEMA_PATH,
        DICTIONARY_MAP_PATH,
        TRAINING_DATA,
        MODEL_LIST,
        # MODELS_DIR
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Config error: {path} does not exist")


_validate_paths()

DEFAULT_PATTERN_FOR_LIKE_OPERATION = "itis"