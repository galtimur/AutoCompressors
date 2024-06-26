from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

from base_trainer import BaseTrainer
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from utils import calc_grad

from transformers.trainer_utils import EvalPrediction

# from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
from transformers.trainer_utils import (
    EvalPrediction,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    TrainerCallback
)
from transformers.training_args import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


logger = logging.get_logger(__name__)

def convert_past_kv_bfloat_and_detach(past_kv):
    if past_kv is None:
        return None
    return tuple([(past_kv_layer[0].detach().bfloat16(), past_kv_layer[1]) for past_kv_layer in past_kv])

def copy_segment(input_ids, segment_size: int, n_to_copy: int):
    s = segment_size
    n = n_to_copy
    input_ids[:, n * s:(n + 1) * s] = input_ids[:, -s:]

    return input_ids

class DataCollator:
    """Simple data collator for language modeling with padding."""
    def __init__(self, tokenizer, additional_args):
        self.tokenizer = tokenizer
        self.additional_args = additional_args
        self.pad_token_id = self.tokenizer.bos_token_id
        self.counter = 0
        self.segment_to_copy = 0

    def __call__(self, features: Any) -> Dict[str, Any]:
        self.counter += 1
        bsz = len(features)
        max_length = max(len(feature["input_ids"]) for feature in features)
        # max_length = self.max_length

        input_ids = torch.full((bsz, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(bsz, max_length, dtype=torch.long)
        labels = torch.full((bsz, max_length), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            input_ids[i, :len(feature["input_ids"])] = torch.tensor(feature["input_ids"], dtype=torch.long)
            if "attention_mask" in feature:
                attention_mask[i, :len(feature["input_ids"])] = torch.tensor(feature["attention_mask"], dtype=torch.long)
            else:
                # TODO account for padding
                attention_mask = torch.ones_like(input_ids)
            if "labels" in feature:
                labels[i, :len(feature["input_ids"])] = torch.tensor(feature["labels"], dtype=torch.long)
            else:
                # TODO account for padding
                labels[i, : len(feature['input_ids'])] = input_ids[i, :len(feature['input_ids'])]

        # if self.counter % 6000 == 0:
        #     self.segment_to_copy += 1
        #     print(self.segment_to_copy)
        # if self.segment_to_copy < 5:
        #     input_ids = copy_segment(input_ids, 1024, self.segment_to_copy)
        #     labels = copy_segment(labels, 1024, self.segment_to_copy)
        # input_ids = copy_segment(input_ids, 1024, 0)
        # labels = copy_segment(labels, 1024, 0)

        return dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels)


class SubstepTrainer(BaseTrainer):
    """Trainer that implements gradient detaching and accumulating after substeps"""
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(model,
                         args,
                         DataCollator(tokenizer, args),
                         train_dataset,
                         eval_dataset,
                         tokenizer,
                         model_init,
                         compute_metrics,
                         callbacks,
                         optimizers,
                         preprocess_logits_for_metrics)

        self.current_block = 0
        self.loss_log = {f"substep_{i}": 0 for i in range(self.args.training_substeps)}
        self.substep_count = torch.tensor([0])
        self.log_count = 0

    def add_metrics(self, metrics, log_p, labels, prefix=""):
        """Adds metrics to the metrics dictionary. """

        mask = (labels != -100).float()
        nlls = -log_p.gather(-1, labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
        # if mask.sum(-1) == 0:
        #     # This deals with cases of empty segments due to padding, but can lead to inaccurate logging
        #     metrics[f"{prefix}nll"] = torch.tensor(0.0, device=log_p.device)
        #     metrics[f"{prefix}acc"] = torch.tensor(0.0, device=log_p.device)
        # else:
        #     metrics[f"{prefix}nll"] = (nlls * mask).sum(-1) / mask.sum(-1)
        #     correct = (log_p.argmax(-1) == labels).float()
        #     metrics[f"{prefix}acc"] = (correct * mask).sum(-1) / mask.sum(-1)
            
        metrics[f"{prefix}nll"] = (nlls * mask).sum(-1) / mask.sum(-1)
        correct = (log_p.argmax(-1) == labels).float()
        metrics[f"{prefix}acc"] = (correct * mask).sum(-1) / mask.sum(-1)
        return metrics

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        return_output_and_metrics=False
        ):
        """Computes the loss in terms of training blocks. This function is only used during evaluation"""

        total_loss = 0
        softprompt = None
        past_key_values = None
        metrics = {}
        for substep in range(self.args.training_substeps):
            input_slice, segment_lengths = self.segment_input(inputs, substep)
            if torch.any((input_slice["labels"] != -100).sum(-1) == 0):
                continue
            if os.getenv("FA_EVAL", False):         
                out = model(**inputs, segment_lengths=sum(self.args.segment_lengths), use_cache=False)
                softprompt = None
            else:
                out = model(**inputs, softprompt=softprompt, past_key_values=past_key_values, segment_lengths=segment_lengths,
                      use_cache=self.args.use_kv, output_softprompt=True)
                softprompt = out.softprompt
                past_key_values = out.past_key_values["past_key_values"]
            loss = out.loss
            total_loss += loss

            if return_output_and_metrics:
                labels = input_slice["labels"][:, 1:]
                log_p =  out.logits[:, :-1, :].log_softmax(dim=-1)
                self.add_metrics(metrics, log_p, labels, prefix=f"substep_{substep}-avg-")

                # num_segments = math.ceil(input_slice["input_ids"].shape[-1]/self.args.segment_length)
                num_segments = self.args.segments_per_substep
                for i in range(num_segments):
                    start = sum(segment_lengths[:i])
                    end = sum(segment_lengths[:i+1])
                    self.add_metrics(metrics, log_p[:, start:end], labels[:, start:end],
                                     prefix=f"substep_{substep}-seg{i}-")
                    metrics[f"substep_{substep}-seg{i}-numtokens"] = (labels[:, start:end] != - 100).sum(-1)

        if return_output_and_metrics:
            return (total_loss, out, metrics)

        elif return_outputs:
            return (total_loss, out)
        else:
            return total_loss

    def training_substep(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        softprompt: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        segment_lengths = None
        ) -> torch.Tensor:
        """Performs a training substep, after which softprompts are detached and gradients are accumulated"""

        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            out = model(**inputs, softprompt=softprompt, past_key_values=past_key_values, segment_lengths=segment_lengths, use_cache=self.args.use_kv, output_softprompt=True)
            loss = out.loss
            # TODO fix it properly not to move bf16-fp32-bf16 every time
            softprompt = out.softprompt.detach().bfloat16()
            past_key_values = convert_past_kv_bfloat_and_detach(out.past_key_values["past_key_values"])
        # TODO check that we pass n_gpu properly.
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        # if True or self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        if self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward() # TODO rewrite using accelerate

        return loss.detach(), softprompt, past_key_values

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """One training step consists of many training_substeps.

        Note that gradient_accumulation_steps is still measured in full training steps,
        although substeps also implicitly accumulated gradient."""

        total_loss = 0
        softprompt = None
        past_key_values = None
        for substep in range(self.args.training_substeps):
            input_slice, segment_lengths = self.segment_input(inputs, substep)
            loss, softprompt, past_key_values = self.training_substep(model, input_slice, softprompt, past_key_values, segment_lengths)
            total_loss += loss
            self.loss_log[f"substep_{substep}"] += loss
            self.substep_count+=1

        if self.args.training_substeps > 0:
            self.log_count+=1
            if self.log_count % self.args.gradient_accumulation_steps == 0:
                self.substep_count = self.substep_count.to(loss.device)
                self.loss_log["total_substeps"] = self._nested_gather(self.substep_count).sum().item()
                grad_norm = calc_grad(model)
                self.loss_log["grad_norm"] = grad_norm
                for i in range(self.args.training_substeps):
                    self.loss_log[f"substep_{i}"] = self._nested_gather(self.loss_log[f"substep_{i}"]).mean().item()
                self.log(self.loss_log)
                for i in range(self.args.training_substeps):
                    self.loss_log[f"substep_{i}"] = 0
                self.log_count = 0

        return total_loss / self.args.training_substeps

    def random_segment_lengths_old(self, input_ids, num_segments, min_segment_length=5):
        """Returns a list of random segment lengths that sum up to num_segments"""
        max_positions = self.model.config.max_position_embeddings
        if num_segments > 1:
            min_segment_length = max(math.ceil((input_ids.size(1) - max_positions) / (num_segments - 1)), 2)

            total_variable_length = input_ids.size(1) - min_segment_length * num_segments
            if num_segments - 1 > total_variable_length:
                raise ValueError(f"The specified number of segments_per_substep cannot cover the entire input sequence.")
            # This is brain fucking. More understadable is randint:
            # breakpoints = torch.randint(total_variable_length, size = num_segments - 1)
            breakpoints = torch.multinomial(torch.ones(total_variable_length), num_segments - 1)
            segment_lengths = torch.diff(breakpoints.sort(-1).values,
                                        prepend=torch.tensor([0]),
                                        append=torch.tensor([total_variable_length]))
            segment_lengths = (segment_lengths + min_segment_length).tolist()
        else:
            segment_lengths = [input_ids.size(1)]
        return segment_lengths

    def random_segment_lengths(self, input_ids, num_segments, min_segment_length=20):

        """Returns a list of random segment lengths that sum up to num_segments
        std: standard deviation of the segments length in terms of ratio of the average segment length
        """
        std = self.args.randomize_std
        if num_segments > 1:

            total_variable_length = input_ids.size(1) - min_segment_length * num_segments
            if num_segments - 1 > total_variable_length:
                raise ValueError(f"The specified number of segments_per_substep cannot cover the entire input sequence.")

            av_size = total_variable_length/num_segments
            window = av_size*std

            # segment_lengths = np.random.normal(av_size, std, num_segments)
            segment_lengths = np.random.uniform(av_size - window, av_size + window, num_segments)
            excess = (sum(segment_lengths) - total_variable_length) / num_segments
            segment_lengths = np.round(segment_lengths - excess).astype(int)
            excess = sum(segment_lengths) - total_variable_length
            segment_lengths[-1] -= excess
            segment_lengths = (segment_lengths + min_segment_length).tolist()
        else:
            segment_lengths = [input_ids.size(1)]
        return segment_lengths


    def segment_input(self, inputs, substep):
        """Returns the sliced inputs and the random segment lengths when randomize_substeps=True"""
        
        # if using segment_lenghts, keep only the end segment of the inputs.
        # This is useful for evaluation. During training, segment lengths should sum to the total block_size
        if not self.args.randomize_substeps:
            total_length = sum(self.args.segment_lengths) * self.args.training_substeps
            inputs["input_ids"] = inputs["input_ids"][:, -total_length:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -total_length:]
            else:
                # TODO account for padding
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            if "labels" in inputs:
                inputs["labels"] = inputs["labels"][:, -total_length:]
            else:
                # TODO account for padding
                inputs["labels"] = inputs["input_ids"][:, -total_length:]

        slices = torch.linspace(0, inputs["input_ids"].shape[-1], steps=self.args.training_substeps + 1, device=inputs["input_ids"].device, dtype=torch.long)
        input_slice = {k: v[:, slices[substep]: slices[substep+1]] for k, v in inputs.items()}
        if self.args.randomize_substeps:
            segment_lengths = self.random_segment_lengths(input_slice["input_ids"], self.args.segments_per_substep)
        else:
            segment_lengths = self.args.segment_lengths

        return input_slice, segment_lengths
