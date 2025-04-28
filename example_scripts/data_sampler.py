import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT))
from utils.dataset_creator import MimicSchema

schema = MimicSchema()

schema.sample_size = 10
schema.check_status()

# to generate sample values if default file is not there
sample_data = schema.generate_sample_values(save_to="sample_data_demo.json")
print(sample_data)