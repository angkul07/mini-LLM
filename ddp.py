import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        os.environ['RANK'] = os.environ.get('LOCAL_RANK', '0')
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())

def init_ddp(rank, world_size) -> None:
    """Initialize the distributed environment."""

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def is_main_process(rank) -> bool:
    return rank == 0

def ddp_sampler(dataset, world_size, rank, shuffle=True, drop_last=False):
    """Prepare the DataLoader for distributed training."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return sampler

def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()