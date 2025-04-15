import os
from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "mimic_data"

MYSQL_DB_PATH: Final[Path] = DATA_DIR / "mimic4.db"
MIMIC_SCHEMA_PATH: Final[Path] = DATA_DIR / "mimic.json"
DICTIONARY_MAP_PATH: Final[Path] = DATA_DIR / "dictionary.json"

OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "")

MIMIC_SAMPLE_DIR = PROJECT_ROOT / "data" / "custom_dataset" / "sample_data"
QUERY_SAMPLE_DIR = PROJECT_ROOT / "data" / "custom_dataset" / "sample_query_sets"
MIMIC_SAMPLE_PATH = MIMIC_SAMPLE_DIR / "default.json"


def _validate_paths() -> None:
    """Check if critical files/dirs exist at startup."""
    required_paths = [
        DATA_DIR,
        MYSQL_DB_PATH,
        MIMIC_SCHEMA_PATH,
        DICTIONARY_MAP_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Config error: {path} does not exist")


def _validate_api_key() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment variables")


_validate_paths()
_validate_api_key()

DEFAULT_PATTERN_FOR_LIKE_OPERATION = "itis"
