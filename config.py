import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
HINMIX_SAMPLE_TXT = DATA_DIR / "hinmix_sample.txt"
DATASET_PATH = DATA_DIR / "dataset_hf" 
TOKENIZER_MODEL_PREFIX = MODEL_DIR / "hinglish_32k"
TOKENIZER_MODEL_FILE = TOKENIZER_MODEL_PREFIX.with_suffix(".model")
HINDI_DISCOURSE_SCRIPT_PATH = PROJECT_ROOT / "hindi_discourse.py"

# Model paths
MODEL_CHECKPOINT_SAVE_PATH = MODEL_DIR / "model_checkpoint.pth"
FINAL_MODEL_SAVE_PATH = MODEL_DIR / "final_model_state.pt"

# Dataset Parameters: define the dataset size
HINMIX_LCSALIGN_HICM_TAKE = 7000    # take first 7000 samples from the dataset
HINMIX_LCSALIGN_EN_TAKE = 2500
HINMIX_LCSALIGN_HI_TAKE = 2500
TINYSTORIES_TAKE = 10000

# Model Configuration
GPT_CONFIG = {
    "vocab_size": 32000,
    "context_length": 128,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.2,
    "qkv_bias": False,
    "flash": False,
}

# Training Parameters
BATCH_SIZE = 16
NUM_EPOCHS = 1
EVAL_FREQ = 5
EVAL_ITER = 5
LEARNING_RATE = 0.001 
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
INITIAL_LR_RATIO = 0.03
MIN_LR_RATIO = 0.001
GRAD_CLIP_NORM = 1.0

# Sampling Parameters
SAMPLE_MAX_NEW_TOKENS = 50
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_K = 50

# Special Tokens: change according to your dataset
USER_DEFINED_SYMBOLS = ['[HI]', '[EN]']

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)