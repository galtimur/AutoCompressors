from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
import numpy as np
from tokenizers import Tokenizer
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from auto_compressor import LlamaAutoCompressorModel


def equal_size_splits(text_or_token_ids: str | Tensor,
                      split_size: int,
                      tokenizer: Tokenizer | None = None) -> list[int]:
    if isinstance(text_or_token_ids, str):
        assert tokenizer is not None
        text = text_or_token_ids
        token_ids = tokenizer.encode(text, return_tensors='pt')
    else:
        token_ids = text_or_token_ids
    if token_ids.ndim > 1:
        n_toks = token_ids.shape[1]
    else:
        n_toks = len(token_ids)
    num_splits = (n_toks - 1) // split_size + 1
    splits = [split_size] * num_splits
    last_split_rem = n_toks % split_size
    if last_split_rem:
        splits[-1] = last_split_rem
    return splits
 

def evaluate_ppl_red_pajamas(model_or_path: nn.Module | str | Path,
                             ds,
                             batch_size: int,
                             max_samples: int = 100,
                             split_size: int = 1024,
                             disable_tqdm: bool = True
                             ) -> dict[str, float]:
    if isinstance(model_or_path, (str, Path)):
        device = torch.device('cuda')
        model = LlamaAutoCompressorModel.from_pretrained(model_or_path,
                                                         torch_dtype=torch.bfloat16,
                                                         device_map=device)
    else:
        model = model_or_path
        device = model.device

    model_training = model.training
    model.eval()
    # ds = load_dataset('awettig/RedPajama-combined-15B-6K-llama', split='test')
    log_losses = defaultdict(float)
    token_counts = defaultdict(int)
    
    def collate_fn(batch):
        cll_batch = dict()
        for k in batch[0].keys():
            cll_batch[k] = torch.stack([torch.tensor(s[k], device=device) for s in batch])
        return cll_batch
    
    dl = DataLoader(ds, batch_size, collate_fn=collate_fn)
    
    samples_seen = 0
    for samp_num, sample in tqdm(enumerate(dl), disable=disable_tqdm):
        # inp = ds[0]['input_ids']
        inp_ids = sample['input_ids']
        # inp_ids = torch.tensor(inp, dtype=torch.long, device=device).unsqueeze(0)
        split_sizes = equal_size_splits(inp_ids, split_size)
        torch.cuda.empty_cache()
        with torch.no_grad():
            logits = model(inp_ids, segment_lengths=split_sizes).logits
            bs, seq_len = inp_ids.shape
            logits = logits[:, :-1].reshape(-1, logits.shape[2])
            targets = inp_ids[:, 1:].reshape(-1)
            xent = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
            xent = xent.reshape(bs, seq_len - 1)
            chunk_token_loss_list = torch.split(xent, split_size, dim=1)
            for n, token_loss_tens in enumerate(chunk_token_loss_list):
                log_losses[f'chunk_{n}'] += token_loss_tens.sum().item()
                token_counts[f'chunk_{n}'] += token_loss_tens.numel()
            log_losses['total'] += xent.sum().item()
            token_counts['total'] += xent.numel()
        samples_seen += bs
        if 0 < max_samples < samples_seen:
            break
    losses_dict = {}
    for k, v in log_losses.items():
        loss = v / token_counts[k]
        losses_dict[k + '_loss'] = loss
        losses_dict[k + '_ppl'] = np.exp(loss)
        losses_dict[k + '_num_tokens'] = token_counts[k]
        losses_dict['num_samples'] = batch_size * samp_num

    if model_training:
        model.train()

    return losses_dict


def evaluate_base_model(model, dataset, batch_size, max_samples, context_size) -> dict[str, float]:
    model_training = model.training
    model.eval()

    def collate_fn(batch):
        cll_batch = dict()
        for k in batch[0].keys():
            cll_batch[k] = torch.stack([torch.tensor(s[k], device="cuda") for s in batch])
        return cll_batch

    dl = DataLoader(dataset, batch_size, collate_fn=collate_fn)

    samples_seen = 0
    total_loss = 0
    for samp_num, sample in tqdm(enumerate(dl, start=1)):
        inp_ids = sample['input_ids'][:,-context_size:]
        with torch.no_grad():
            loss = model(inp_ids, labels=inp_ids).loss.item()
            total_loss += loss
        samples_seen += batch_size
        if 0 < max_samples < samples_seen:
            break

    if model_training:
        model.train()

    return total_loss/samp_num
