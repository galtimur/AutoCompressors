import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import defaultdict

device = "cuda:0"

# model
model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to(device)

PastKVType = tuple[tuple[torch.FloatTensor]]


def split_even(batch, size):
    batch_split = torch.split(batch, size, dim=1)

    return batch_split


def cut_past_kv(past_kv: PastKVType | None, length: int, seq_first_and_merged_kv=False):
    if past_kv is None:
        return None
    if seq_first_and_merged_kv:
        past_kv = tuple(
            [(past_kv_layer[0][:, -length:], length) for past_kv_layer in past_kv]
        )
    else:
        past_kv = tuple(
            [
                (past_kv_layer[0][:, :, -length:], past_kv_layer[1][:, :, -length:])
                for past_kv_layer in past_kv
            ]
        )
    return past_kv


# Data
def collate_fn(batch):
    batched_tensors = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            batched_tensors[key].append(torch.tensor(value))
    for key, value in batched_tensors.items():
        batched_tensors[key] = torch.stack(batched_tensors[key]).to(device)
    return batched_tensors


def accum_past_kv(
    past_key_values: PastKVType | None,
    new_past_kv: PastKVType | None,
    seq_first_and_merged_kv=False,
):
    if past_key_values is None:
        return new_past_kv

    if seq_first_and_merged_kv:
        agg_dim = 1
    else:
        agg_dim = 2

    past_key_values_agg = []
    for past_kv_layer, new_past_kv_layer in zip(past_key_values, new_past_kv):
        if seq_first_and_merged_kv:
            new_len = past_kv_layer[-1] + new_past_kv_layer[-1]
            past_kv_layer_agg = torch.cat(
                [past_kv_layer[0], new_past_kv_layer[0]], dim=agg_dim
            )
            past_key_values_agg.append((past_kv_layer_agg, new_len))
        else:
            past_k_layer_agg = torch.cat(
                [past_kv_layer[0], new_past_kv_layer[0]], dim=agg_dim
            )
            past_v_layer_agg = torch.cat(
                [past_kv_layer[1], new_past_kv_layer[1]], dim=agg_dim
            )
            past_key_values_agg.append((past_k_layer_agg, past_v_layer_agg))

    return tuple(past_key_values_agg)


train_dataset = load_dataset("awettig/RedPajama-combined-15B-6K-llama", split="test")
train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Number of parameters = {total_params:.0f}")

example = next(iter(train_dataloader))
input_ids = torch.tensor(example["input_ids"])

# TODO pass split size from config
input_ids_splitted = split_even(input_ids, 1024)
past_kv = None
for split in input_ids_splitted:
    input_embeds = model.base_model.embed_tokens(split)
    out = model(
        inputs_embeds=input_embeds,
        labels=split,
        output_hidden_states=True,
        use_cache=True,
        past_key_values=past_kv,
    )
    new_past_kv = cut_past_kv(out.past_key_values, 50, seq_first_and_merged_kv=False)
    past_kv = accum_past_kv(past_kv, new_past_kv, seq_first_and_merged_kv=False)
    print(past_kv[0][0].shape)

pass
