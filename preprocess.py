import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tokenizer import Tokenizer

class TextChunkDataset(TorchDataset):
    def __init__(self, texts: list[str] | str, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            if not text:
                continue
            token_ids = tokenizer.encode(text)
                
            # Create chunks of input and target sequences
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i : i + max_length]
                target_chunk = token_ids[i + 1 : i + max_length + 1]

                if len(input_chunk) == max_length and len(target_chunk) == max_length:
                    self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                    self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]
    
def create_dataloader(
    dataset,
    batch_size: int, 
    shuffle: bool = True, 
    drop_last: bool = True, 
    sampler=None
) -> DataLoader:
    

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(sampler is None and shuffle), 
        drop_last=drop_last, 
        sampler=sampler,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader

def text_to_token_ids(text: str, tokenizer: Tokenizer, device: torch.device | str) -> torch.Tensor:
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    return encoded_tensor

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Tokenizer) -> str:
    if token_ids.ndim > 1:
        token_ids = token_ids.squeeze(0)
    return tokenizer.decode(token_ids.tolist())