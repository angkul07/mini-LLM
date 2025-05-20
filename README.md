# MINI-LLM

A minimal PyTorch implementation for training your own small LLM from scratch. Designed for educational purposes and simplicity, featuring core transformer components and a basic training loop. This specific LLM is trained on a small bilingual of dataset of hindi and english which is tokenized using a custom tokenizer. This can generate text in Hindi and English.

## Features

- **Minimal Codebase**: Pure PyTorch implementation.
- **Core Architecture**: Standard GPT model components:
  - Multi-Head Self-Attention (with optional Flash Attention support if available)
  - Layer Normalization
  - Feed-Forward Network with GELU activation
  - Standard Positional Embeddings
- **Training Features**:
  - Learning rate decay with warmup.
  - Weight decay & gradient clipping.
  - Distributed Data Parallel (DDP) support for multi-GPU training.
- **Dataset Support**: Integration with HuggingFace datasets (demonstrated with HINMIX and TinyStories) and custom text chunking.
- **Custom Tokenizer**: SentencePiece tokenizer training integration.

## Installation 

Clone the repository (assuming it's in a git repo, otherwise just save the files):

```bash
# git clone <your-repo-url>
# cd smol-gpt
pip install -r requirements.txt
```

## Requirements:

- Python 3.8+
- PyTorch 2.0+ (with CUDA recommended for training)
- Modern GPU (recommended for training)

## Outputs:
```markdown
Starting prompt: 'The dragon flew over the mountains'

Generated text:
The dragon flew over the mountains, and he saw many people in the sea. He was so excited ⁇  He played in the ocean 
and explored all day long. ⁇ Suddenly, he heard a voice calling out, "Hey ⁇  Joe ⁇  What are you doing here?" ⁇ 
The dragon looked around and saw a big fish. The fish said, "We are having a race to catch fish ⁇ " ⁇ Joe was 
embarrassed. He said, "But ⁇ " ⁇ So the dragon said, "Let me show you something ⁇ " So Joe quickly ran to the
port and started running around. There were fish, fish and fish and rabbits. Joe stopped rushing and looked 
at the dragon. He said, "There you are little fish. You are so brave ⁇ " ⁇ The dragon smiled and said, "I 
have a great race ⁇  Now I can race ⁇ " Joe smiled and said, "Ok, let's race ⁇ " ⁇ Joe was excited and he ran 
forward. He ran as fast as he could. He crossed the finish line first, and he was laughing. After a while, he 
crossed the finish line again and he crossed the finish line first before it was time to go home.
```

## Quick Start

This guide walks you through the full training cycle.


#### 1. Prepare Dataset

Run the script to download/process the dataset and prepare the tokenizer training file (`hinmix_sample.txt`) and the main training dataset (`dataset_hr`).

```bash
python Datasets.py
```

> **Datasets downloaded are:**
> 
> **For training tokenizer:**  
> A combined dataset of `lcsalign-hicm`, `lcsalign-en`, and `lcsalign-hi` variants from the [HINMIX_hi-en](https://huggingface.co/datasets/kartikagg98/HINMIX_hi-en) dataset:  
> - `lcsalign-hicm`: आपकी Car में black box?  
> - `lcsalign-en`: A black box in your car?  
> - `lcsalign-hi`: आपकी कार में ब्लैक बॉक्स?
> 
> **For LLM training:**  
> A combined dataset of Hindi stories from [Hindi Discourse](https://huggingface.co/datasets/midas/hindi_discourse) and a subset of the [Tiny Stories](https://huggingface.co/roneneldan/TinyStories-1M) dataset.


#### 2. Train Tokenizer

Train the SentencePiece tokenizer using the prepared sample file. This will create the `.model` and `.vocab` files specified in `config.py`.

```bash
python train_tokenizer.py
```


#### 3. Start Training

Run the main training script. This will load the dataset, initialize the model, and start the training loop. Checkpoints will be saved periodically.

For DDP training on multiple GPUs, use the `torchrun` command (e.g., for 2 GPUs):

```bash
# Single GPU/CPU
python train.py

# Multi-GPU (example for 2 GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```

*Training and validation loss progress will be printed to the console. A loss plot will be saved as `loss_plot.png` after training.*


#### 4. Generate Text

Once training is complete and a checkpoint (`model_checkpoint.pth` or `final_model_state.pt`) is saved, you can use the `sample.py` script. This script currently uses the checkpoint and configuration paths defined in `config.py`.

```bash
python sample.py
```

> **Note:** In the `sample.py` change the `start_prompt` to generate different texts.

## Use Pre-trained model(if not want to train)
#### 1. Download tokenizer and model
```bash
# Download tokenizer
wget https://huggingface.co/OmAlve/TinyStories-SmolGPT/resolve/main/tok4096.model -P data/

# Download pre-trained checkpoint
wget https://huggingface.co/OmAlve/TinyStories-SmolGPT/resolve/main/ckpt.pt -P model/
```

#### 2. Run inference
```bash
python sample.py
```

## Configuration
Key parameters can be modified in the `config.py` file:

#### Dataset Sampling Sizes:
```python
# Example Sampling Sizes
HINMIX_LCSALIGN_HICM_TAKE = 7000
HINMIX_LCSALIGN_EN_TAKE = 2500
HINMIX_LCSALIGN_HI_TAKE = 2500
TINYSTORIES_TAKE = 10000
```
>**Note:** Change these parameters to change the size of the dataset. This dataset is very small for llm training

#### Model Architecture (GPT_CONFIG):
```python
GPT_CONFIG = {
    "vocab_size": 32000,        # Vocabulary size (should match tokenizer)
    "context_length": 128,      # Maximum sequence length
    "emb_dim": 256,             # Embedding dimension
    "n_heads": 4,               # Number of attention heads
    "n_layers": 4,              # Number of transformer layers
    "drop_rate": 0.2,           # Dropout rate
    "qkv_bias": False,          # Use bias in QKV linear layers
    "flash": False,             # Enable Flash Attention (requires compatible hardware/PyTorch)
}
```
>**Note:** I trained a small model due to the limited compute and data. To train a tokenizer with differnet vocab size, change the vocab_size value.

#### Training Parameters:
```python
# Example Training Parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
EVAL_FREQ = 5           # Evaluate loss every N global steps
EVAL_ITER = 5           # Number of batches to use for evaluation
LEARNING_RATE = 0.01    # Peak learning rate
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1      # Percentage of steps for linear LR warmup
INITIAL_LR_RATIO = 0.03 # Start LR is INITIAL_LR_RATIO * LEARNING_RATE
MIN_LR_RATIO = 0.001    # Minimum LR is MIN_LR_RATIO * LEARNING_RATE
```

## File structure:
```bash
smol-gpt/
├── config.py           - Model & training configuration
├── Datasets.py         - Data loading and preparation scripts
├── ddp.py              - DDP setup utilities
├── model.py            - GPT model implementation
├── plot.py             - Utility for plotting losses
├── preprocess.py       - Dataset chunking and token/text conversion
├── sample.py           - Text generation script
├── tokenizer.py        - Tokenizer wrapper
├── train.py            - Main training loop
└── train_tokenizer.py  - Script for training the SentencePiece tokenizer
```

> **Note:** Every parameter in `config.py` can be changed according to requirements :)

## Contributions and suggestions are welcome :)