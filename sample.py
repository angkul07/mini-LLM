import torch
from model import GPTModel
from tokenizer import Tokenizer
from preprocess import text_to_token_ids, token_ids_to_text
import config

def generate_text(
    model: GPTModel, 
    tokenizer: Tokenizer,
    start_text: str, 
    max_new_tokens, 
    context_size, 
    device: torch.device | str,
    temperature = 1.0, 
    top_k: int | None = None,
    eos_id: int | None = None,
    train_mode = False
) -> str:
    model.eval()

    idx = text_to_token_ids(start_text, tokenizer, device)

    for _ in range(max_new_tokens):
        # Crop context if current sequence is longer than context_size
        idx_cond = idx if idx.size(1) <= context_size else idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        
        logits = logits[:, -1, :]

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            if v.numel() > 0:
                 min_val_to_keep = v[:, -1].unsqueeze(-1)
                 logits = torch.where(logits < min_val_to_keep, torch.tensor(float("-inf")).to(logits.device), logits)
            else:
                pass

        # Apply temperature scaling
        if temperature > 0.0:
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else: # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    if train_mode:
        model.train()
    return token_ids_to_text(idx, tokenizer)


if __name__ == "__main__":
    device_str = config.DEVICE
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    try:
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        model_cfg = checkpoint.get('model_config', config.GPT_CONFIG) # Get config from checkpoint or default
        
        loaded_model = GPTModel(config.GPT_CONFIG) # Use the config used for training
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
    

        loaded_model.to(device)
        loaded_model.eval()

        tokenizer_instance = Tokenizer()

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