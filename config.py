import os

# Model configurations
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Directory paths
DATA_DIR = "data"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# RAG configurations
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 3

# LLM configurations
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
TOP_P = 0.9
DO_SAMPLE = True

# Device configuration
DEVICE = "cuda" if os.path.exists("/usr/local/cuda") else "cpu"

# Evaluation metrics
EVAL_METRICS = ["f1_score", "bleu", "rouge"]