import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from .tokenizer import CharTokenizer

class TextDataset(Dataset):
    """
    Dataset that creates contiguous sequences with shifted targets for next-token prediction.
    Given sequence tokens:
        input x:  tokens[i : i + context_length]
        target y: tokens[i + 1 : i + context_length + 1]
    """
    def __init__(self, token_ids: list[int] | torch.Tensor, context_length: int):
        if not isinstance(token_ids, torch.Tensor):
            self.data = torch.tensor(token_ids, dtype=torch.long)
        else:
            self.data = token_ids.to(dtype=torch.long)
            
        self.context_length = context_length
        self.num_samples = max(0, len(self.data) - context_length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


def prepare_datasets(
    text_path: str,
    context_length: int = 128,
    train_split: float = 0.9,
    tokenizer: Optional[CharTokenizer] = None
) -> Tuple[TextDataset, TextDataset, CharTokenizer]:
    """
    Reads corpus file, trains/loads tokenizer, splits into train/validation,
    and returns TextDatasets and Tokenizer.
    """
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    if tokenizer is None:
        tokenizer = CharTokenizer.train_from_text(text)

    token_ids = tokenizer.encode(text)
    split_idx = int(len(token_ids) * train_split)

    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    print(f"[Dataset] Total tokens: {len(token_ids):,} | Train: {len(train_tokens):,} | Val: {len(val_tokens):,}")

    train_dataset = TextDataset(train_tokens, context_length)
    val_dataset = TextDataset(val_tokens, context_length)

    return train_dataset, val_dataset, tokenizer


def get_batch_fast(data_tensor: torch.Tensor, context_length: int, batch_size: int, device: str = "cpu"):
    """
    Fast batch sampler using random offsets (similar to Karpathy nanoGPT style).
    Useful for quick training iterations and evaluations.
    """
    ix = torch.randint(len(data_tensor) - context_length, (batch_size,))
    x = torch.stack([data_tensor[i : i + context_length] for i in ix])
    y = torch.stack([data_tensor[i + 1 : i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)
