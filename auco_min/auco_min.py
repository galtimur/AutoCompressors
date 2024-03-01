import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AcCausalLMOutputWithPast(CausalLMOutputWithPast):
    summary_vectors: list[Tensor] | None = None



PastKVType = tuple[tuple[torch.FloatTensor]]


class ACCausalLM(LlamaForCausalLM):
    def __init__(
        self,
        *args,
        substeps,
        segments_per_substep,
        num_summary_vectors: int = 50,
        split_size=1024,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        emb_dim = self.config.hidden_size
        self.num_summary_vectors = num_summary_vectors
        self.split_size = split_size
        self.substeps = substeps
        self.segments_per_substep = segments_per_substep
        self.summary_embeddings = nn.Embedding(num_summary_vectors, emb_dim)
        self.pad_label = -100
        self.loss_log = defaultdict(int)

    def split_even(self, batch, size):
        batch_split = torch.split(batch, size, dim=1)

        return batch_split

    def cut_past_kv(
        self, past_kv: PastKVType | None, length: int, seq_first_and_merged_kv=False
    ):
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

    def accum_past_kv(
        self,
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

    def convert_past_kv_bfloat_and_detach(self, past_kv):
        if past_kv is None:
            return None
        return tuple([(past_kv_layer[0].detach().bfloat16(), past_kv_layer[1].detach().bfloat16()) for past_kv_layer in past_kv])

    def forward_segment(self, split: torch.LongTensor, past_kv: PastKVType, summary_tok_embs) -> tuple[PastKVType]:

        pad_input = (self.pad_label * torch.ones_like(summary_tok_embs[:, :, 0])).int().to(self.device)
        input_embeds = self.model.embed_tokens(split)
        input_embeds = torch.cat([input_embeds, summary_tok_embs], 1)
        split = torch.cat([split, pad_input], dim=1)

        out = super().forward(
            inputs_embeds=input_embeds,
            labels=split,
            use_cache=True,
            past_key_values=past_kv,
        )
        new_past_kv = self.cut_past_kv(
            out.past_key_values,
            self.num_summary_vectors,
            seq_first_and_merged_kv=False,
        )
        past_kv = self.accum_past_kv(
            past_kv, new_past_kv, seq_first_and_merged_kv=False
        )

        return out, past_kv

    def forward(
        self, input_ids: torch.LongTensor = None, *args, **kwargs
    ) -> tuple | AcCausalLMOutputWithPast:
        batch_size = input_ids.size(0)
        input_ids_splitted = self.split_even(input_ids, 1024)
        assert (
            len(input_ids_splitted) == self.substeps * self.segments_per_substep
        ), f"Number of substeps*segments_per_substep should equal to number of segments in the splitted ids. len(input_ids_splitted) = {len(input_ids_splitted)}, substeps = {self.substeps}, segments_per_substep = {self.segments_per_substep}"

        summary_tok_embs = self.summary_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        past_kv = None
        total_loss = 0
        num_segments = len(input_ids_splitted)
        for split_num, split in enumerate(input_ids_splitted):
            out, past_kv = self.forward_segment(split, past_kv, summary_tok_embs)
            loss_segment = out.loss.item()
            total_loss += loss_segment/num_segments
            if (split_num+1)%self.segments_per_substep==0:
                past_kv = self.convert_past_kv_bfloat_and_detach(past_kv)
                loss = out.loss
                # TODO may be here we need not use accelerate.
                if self.training:
                    loss.backward()

            self.loss_log[f"segment_{split_num}"] = loss_segment
        self.loss_log["loss"] = total_loss
        out.loss_log = self.loss_log

        return out
