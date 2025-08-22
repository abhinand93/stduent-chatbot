import os

# Base dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Config files
APP_CONFIG_FPATH = os.path.join(BASE_DIR, "config", "config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(BASE_DIR, "config", "prompt_config.yaml")

# Vector DB directory (single consistent path!)
VECTOR_DB_DIR = os.path.join(BASE_DIR, "student_knowledge_base", "vectorstore")
