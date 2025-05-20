import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def setup(rank: int, world_size: int) -> None:
    """Initialize the distributed environment."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)
    torch.distributed.barrier()  # Ensure all processes are synchronized

def prepare(rank, world_size, batch_size, dataset, pin_memory=False, num_workers=0):
    """Prepare the DataLoader for distributed training."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    return sampler

def cleanup() -> None:
    dist.destroy_process_group()