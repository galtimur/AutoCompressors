import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import defaultdict

# Data
def collate_fn_wrapper(device):
    def collate_fn(batch):
        batched_tensors = defaultdict(list)
        for item in batch:
            for key, value in item.items():
                batched_tensors[key].append(torch.tensor(value))
        for key, value in batched_tensors.items():
            batched_tensors[key] = torch.stack(batched_tensors[key]).to(device)
        return batched_tensors

    return collate_fn


def get_dataloader(batch_size:int=1, device: str="cuda:0"):

    collate_fn = collate_fn_wrapper(device)
    train_dataset = load_dataset("awettig/RedPajama-combined-15B-6K-llama", split="test")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return train_dataloader