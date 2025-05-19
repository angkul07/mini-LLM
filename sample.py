import torch
from model import GPTModel # Assuming GPTModel is in model.py
from tokenizer import Tokenizer
from preprocess import text_to_token_ids, token_ids_to_text
import config

def generate_text(
    model: GPTModel, 
    tokenizer: Tokenizer,
    start_text: str, 
    max_new_tokens: int, 
    context_size: int, 
    device: torch.device | str,
    temperature: float = 1.0, 
    top_k: int | None = None,
    eos_id: int | None = None
) -> str:
    """Generates text using the provided model and tokenizer."""
    model.eval() # Set model to evaluation mode

    # Encode the starting text
    idx = text_to_token_ids(start_text, tokenizer, device)

    for _ in range(max_new_tokens):
        # Crop context if current sequence is longer than context_size
        idx_cond = idx if idx.size(1) <= context_size else idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond) # (batch, seq_len, vocab_size)
        
        # Get logits for the last token
        logits = logits[:, -1, :] # (batch, vocab_size)

        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Ensure v has elements before accessing v[:, -1]
            if v.numel() > 0:
                 min_val_to_keep = v[:, -1].unsqueeze(-1) # Make it (batch, 1) for broadcasting
                 logits = torch.where(logits < min_val_to_keep, torch.tensor(float("-inf")).to(logits.device), logits)
            else: # If top_k is 0 or logits are empty for some reason
                pass # No filtering or handle as error

        # Apply temperature scaling
        if temperature > 0.0: # temperature == 0 means greedy decoding
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (batch, 1)
        else: # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (batch, 1)

        # Check for EOS token
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # Append generated token
        idx = torch.cat((idx, idx_next), dim=1)

    # Decode the generated sequence
    return token_ids_to_text(idx, tokenizer)


if __name__ == "__main__":
    device_str = config.DEVICE
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    try:
        # --- Model Loading ---
        # Option 1: Load from a checkpoint (if saved with model_state_dict)
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        model_cfg = checkpoint.get('model_config', config.GPT_CONFIG) # Get config from checkpoint or default
        
        loaded_model = GPTModel(config.GPT_CONFIG) # Use the config used for training
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
    

        loaded_model.to(device)
        loaded_model.eval()

        tokenizer_instance = Tokenizer() # Loads from config.TOKENIZER_MODEL_FILE

        # --- Text Generation ---
        start_prompt = "Once upon a time"
        print(f"Starting prompt: '{start_prompt}'")

        generated_output = generate_text(
            model=loaded_model,
            tokenizer=tokenizer_instance,
            start_text=start_prompt,
            max_new_tokens=config.SAMPLE_MAX_NEW_TOKENS,
            context_size=config.GPT_CONFIG["context_length"],
            device=device,
            temperature=config.SAMPLE_TEMPERATURE,
            top_k=config.SAMPLE_TOP_K,
            eos_id=tokenizer_instance.eos_id
        )
        print("\nGenerated text:")
        print(generated_output)

    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure model and tokenizer files exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")