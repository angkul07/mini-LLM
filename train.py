import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import math
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict

from sample import generate_text
from model import GPTModel
from preprocess import create_dataloader
from tokenizer import Tokenizer
from plot import plot_losses
from ddp import setup, cleanup
import config

is_ddp = config.DDP
DEVICE = config.DEVICE

def calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculates cross-entropy loss."""
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())

@torch.no_grad()
def calculate_average_loss(data_loader: torch.utils.data.DataLoader, model: GPTModel, 
                           device: DEVICE, num_batches: int | None = None) -> float:
    """Calculates average loss over a data loader."""
    model.eval()
    total_loss = 0.0
    actual_batches = 0

    if len(data_loader) == 0:
        return float('nan')

    if num_batches is None:
        batches_to_iterate = len(data_loader)
    else:
        batches_to_iterate = min(num_batches, len(data_loader))
    
    if batches_to_iterate == 0:
        return float('nan')

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= batches_to_iterate:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = calculate_loss(logits, target_batch)
        total_loss += loss.item()
        actual_batches += 1
    
    model.train()
    return total_loss / actual_batches if actual_batches > 0 else float('nan')


def get_lr(it: int, warmup_steps: int, total_training_steps: int, 
           peak_lr: float, min_lr: float, initial_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return initial_lr + (peak_lr - initial_lr) * (it / warmup_steps)
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= total_training_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (total_training_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def train_model():
    torch.manual_seed(123)
    device = DEVICE
    setup(rank=config.RANK, world_size=config.WORLD_SIZE) if is_ddp else None  

    try:
        tokenizer = Tokenizer()
        if config.GPT_CONFIG["vocab_size"] != tokenizer.vocab_size:
            print(f"Warning: Config vocab_size ({config.GPT_CONFIG['vocab_size']}) "
                  f"differs from tokenizer vocab_size ({tokenizer.vocab_size}). "
                  f"Using tokenizer's vocab_size for the model.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_tokenizer.py first.")
        return
    
    if is_ddp:
        model = GPTModel(config.GPT_CONFIG).to(config.RANK)
        model = DistributedDataParallel(model, device_ids=[config.RANK], output_device=config.RANK)
    else:
        model = GPTModel(config.GPT_CONFIG).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    try:
        combined_hf_dataset = load_from_disk(str(config.COMBINED_DATASET_PATH))
    except FileNotFoundError:
        print(f"Error: Combined dataset not found at {config.COMBINED_DATASET_PATH}")
        print("Please run Datasets.py (create_combined_dataset) first.")
        return

    if not isinstance(combined_hf_dataset, DatasetDict):
        if hasattr(combined_hf_dataset, 'shuffle'):
            combined_hf_dataset = combined_hf_dataset.shuffle(seed=42)
        split_data = combined_hf_dataset.train_test_split(test_size=0.1, seed=123)
    else:
        split_data = combined_hf_dataset

    train_texts = split_data['train']['text']
    val_texts = split_data['test']['text']

    train_loader = create_dataloader(
        train_texts, tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.GPT_CONFIG['context_length'],
        stride=config.GPT_CONFIG['context_length'] // 2, # 50% overlap: for smaller datasets else stride=config.GPT_CONFIG['context_length']
        shuffle=True, drop_last=True, num_workers=2
    )
    val_loader = create_dataloader(
        val_texts, tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.GPT_CONFIG['context_length'],
        stride=config.GPT_CONFIG['context_length'] // 2,
        shuffle=False, drop_last=False, num_workers=2
    )
    
    if len(train_loader) == 0 or len(val_loader) == 0:
        print("Error: One or both dataloaders are empty. Check dataset processing.")
        return

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, # Peak LR
        weight_decay=config.WEIGHT_DECAY
    )

    total_training_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(config.WARMUP_RATIO * total_training_steps)
    initial_lr = config.LEARNING_RATE * config.INITIAL_LR_RATIO
    min_lr = config.LEARNING_RATE * config.MIN_LR_RATIO
    
    print(f"Total training steps: {total_training_steps}, Warmup steps: {warmup_steps}")
    print(f"Initial LR: {initial_lr:.2e}, Peak LR: {config.LEARNING_RATE:.2e}, Min LR: {min_lr:.2e}")


    #Training Loop --->
    train_losses, val_losses, tracked_lrs, tokens_seen_log = [], [], [], []
    global_step = 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()

        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", unit="batch")
        
        for input_batch, target_batch in epoch_progress:
            # Learning rate scheduling
            lr = get_lr(global_step, warmup_steps, total_training_steps, config.LEARNING_RATE, min_lr, initial_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            tracked_lrs.append(lr)

            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(input_batch)
            loss = calculate_loss(logits, target_batch)
            loss.backward()
            
            # Gradient Clipping (after warmup, or always if preferred)
            if global_step >= warmup_steps:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
            
            optimizer.step()
            
            tokens_seen_log.append(global_step * config.BATCH_SIZE * config.GPT_CONFIG['context_length'])
            global_step += 1

            # Logging and Evaluation
            if global_step % config.EVAL_FREQ == 0:
                avg_train_loss = calculate_average_loss(train_loader, model, device, num_batches=config.EVAL_ITER)
                avg_val_loss = calculate_average_loss(val_loader, model, device, num_batches=config.EVAL_ITER)
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                epoch_progress.set_postfix({
                    "train_loss": f"{avg_train_loss:.3f}", 
                    "val_loss": f"{avg_val_loss:.3f}",
                    "lr": f"{lr:.2e}"
                })

        print(f"\nEpoch {epoch+1} completed.")
        avg_epoch_train_loss = calculate_average_loss(train_loader, model, device)
        avg_epoch_val_loss = calculate_average_loss(val_loader, model, device)
        print(f"End of Epoch {epoch+1} Train Loss: {avg_epoch_train_loss:.3f} Val Loss: {avg_epoch_val_loss:.3f}")

        sample_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            start_text="You are a helpful assistant",
            max_new_tokens=config.SAMPLE_MAX_NEW_TOKENS,
            context_size=config.GPT_CONFIG["context_length"],
            device=device,
            temperature=config.SAMPLE_TEMPERATURE,
            top_k=config.SAMPLE_TOP_K,
            eos_id=tokenizer.eos_id,
            train_mode=True
        )
        print(f"Sample generation: {sample_output.replace(chr(0), '').replace(chr(25917), '')}")

        # --- Save Checkpoint ---
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config.GPT_CONFIG, # Save model config with checkpoint
            "train_loss": avg_train_loss if train_losses else None,
            "val_loss": avg_epoch_val_loss
        }
        torch.save(checkpoint, config.MODEL_CHECKPOINT_SAVE_PATH)
        print(f"Checkpoint saved to {config.MODEL_CHECKPOINT_SAVE_PATH}")

    if is_ddp:
        # Save the model state dict for DDP
        torch.save(model.module.state_dict(), config.MODEL_CHECKPOINT_SAVE_PATH)
        cleanup()

    print("Training complete.")
    # save the final model's weights only for easier deployment/sharing
    torch.save(model.state_dict(), config.FINAL_MODEL_SAVE_PATH)
    print(f"Final model weights saved to {config.FINAL_MODEL_SAVE_PATH}")

    plot_losses(
    train_losses=train_losses,
    val_losses=val_losses,
    tokens_seen_log=tokens_seen_log,
    save_path="loss_plot.png"
    )   

if __name__ == "__main__":
    if is_ddp:
        mp.spawn(train_model, nprocs=config.NUM_GPUS, join=True)
    else:
        train_model()