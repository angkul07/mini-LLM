import os
import math
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.distributed as dist
from datasets import load_from_disk, DatasetDict
from torch.nn.parallel import DistributedDataParallel

import config
from model import GPTModel
from plot import plot_losses
from tokenizer import Tokenizer
from sample import generate_text
from preprocess import create_dataloader, TextChunkDataset
from ddp import setup_ddp, init_ddp, ddp_sampler, cleanup, is_main_process

DEVICE = config.DEVICE

def calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculates cross-entropy loss."""
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())

@torch.no_grad()
def calculate_average_loss(data_loader: torch.utils.data.DataLoader, model: GPTModel, 
                           device= DEVICE, num_batches: int | None = None, is_ddp=False, world_size=1) -> float:
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
        if is_ddp:
            reduced_loss = loss.detach().clone()
            dist.reduce(reduced_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss += reduced_loss.item() / world_size
        else:
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
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    is_ddp = local_rank != -1 and world_size > 1

    if is_ddp:
        setup_ddp()
        init_ddp(local_rank, world_size)
        device = local_rank
    else:
        local_rank = 0
        world_size = 1
        device = DEVICE
        is_ddp = False

    if is_main_process(local_rank):
        print(f"Running on device: {device}. DDP active: {is_ddp}. World size: {world_size}")

    torch.manual_seed(123+local_rank)

    tokenizer = Tokenizer()
    
    model = GPTModel(config.GPT_CONFIG).to(device)

    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # Dataset Loading -->
    dataset_hf = load_from_disk(str(config.DATASET_PATH))

    if not isinstance(dataset_hf, DatasetDict):
        if hasattr(dataset_hf, 'shuffle'):
            dataset_hf = dataset_hf.shuffle(seed=42)
        split_data = dataset_hf.train_test_split(test_size=0.1, seed=123+local_rank)
    else:
        split_data = dataset_hf

    train_texts = split_data['train']['text']
    val_texts = split_data['test']['text']

    train_set = TextChunkDataset(train_texts, tokenizer, config.GPT_CONFIG['context_length'], config.GPT_CONFIG['context_length']//2)
    val_set = TextChunkDataset(val_texts, tokenizer, config.GPT_CONFIG['context_length'], config.GPT_CONFIG['context_length']//2)

    train_sampler = ddp_sampler(train_set, world_size=world_size, rank=local_rank)
    val_sampler = ddp_sampler(val_set, world_size=world_size, rank=local_rank)

    train_loader = create_dataloader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True, drop_last=True,
        sampler=train_sampler
    )
    val_loader = create_dataloader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True, drop_last=True,
        sampler=val_sampler
    )
    
    if len(train_loader) == 0 or len(val_loader) == 0 and is_main_process(local_rank):
        print("Error: One or both dataloaders are empty. Check dataset processing.")
        if is_ddp: 
            cleanup()
        return

    optimizer_model = model.module if is_ddp else model
    optimizer = optim.AdamW(
        optimizer_model.parameters(), 
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
    tokens_seen, global_step = 0, 0

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
            
            # tokens_seen_log.append(global_step * config.BATCH_SIZE * config.GPT_CONFIG['context_length'])
            tokens_seen += input_batch.numel()
            global_step += 1

            # Logging and Evaluation
            if global_step % config.EVAL_FREQ == 0:
                avg_train_loss = calculate_average_loss(train_loader, model, device, num_batches=config.EVAL_ITER, is_ddp=is_ddp, world_size=world_size)
                avg_val_loss = calculate_average_loss(val_loader, model, device, num_batches=config.EVAL_ITER, is_ddp=is_ddp, world_size=world_size)
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                tokens_seen_log.append(tokens_seen)
                
                epoch_progress.set_postfix({
                    "train_loss": f"{avg_train_loss:.3f}", 
                    "val_loss": f"{avg_val_loss:.3f}",
                    "lr": f"{lr:.2e}"
                })

        print(f"\nEpoch {epoch+1} completed.")
        avg_epoch_train_loss = calculate_average_loss(train_loader, model, device, is_ddp=is_ddp, world_size=world_size)
        avg_epoch_val_loss = calculate_average_loss(val_loader, model, device, is_ddp=is_ddp, world_size=world_size)
        print(f"End of Epoch {epoch+1} Train Loss: {avg_epoch_train_loss:.3f} Val Loss: {avg_epoch_val_loss:.3f}")

        actual_model_for_gen = model.module if is_ddp else model
        sample_output = generate_text(
            model=actual_model_for_gen,
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
            "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config.GPT_CONFIG,
            "train_loss": avg_train_loss if train_losses else None,
            "val_loss": avg_epoch_val_loss
        }
        torch.save(checkpoint, config.MODEL_CHECKPOINT_SAVE_PATH)
        print(f"Checkpoint saved to {config.MODEL_CHECKPOINT_SAVE_PATH}")

    if is_main_process(local_rank):
        print("Training complete.")
        final_model_state = {
            "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
            "model_config": config.GPT_CONFIG
        }
        torch.save(final_model_state, str(config.FINAL_MODEL_SAVE_PATH))
        print(f"Final model weights saved to {config.FINAL_MODEL_SAVE_PATH}")
    
    if is_ddp:
        cleanup()

    plot_losses(
    train_losses=train_losses,
    val_losses=val_losses,
    tokens_seen_log=tokens_seen_log,
    save_path="loss_plot.png"
    )   

if __name__ == "__main__":
    train_model()