import os

# Database Config
DB_CONFIG = {
    "host": "localhost",
    "user": "aaryan",
    "password": os.getenv("MIMIC_SQL_PW"),
    "database": "capstone_mimic"
}

# OpenAI API Key (GPT-4)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")