from dataclasses import dataclass
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)


from substep_trainer import SubstepTrainer
from config_parser import parse_config

from data import load_preprocessed_datasets
from fast_attention import patch_opt

@dataclass
class ModelTrainingModules:
    trainer: object
    model_kwargs: dict
    train_dataset: object
    val_dataset: object
    merge_config: object

def get_run_mudules(args, config_path, bench_parameters={}, max_val_samples=300, return_dataset=True, return_trainer=True):

    model_args, data_args, training_args, merge_config = parse_config(config_path, args, override_dict=bench_parameters)
    model_args.use_kv = training_args.use_kv

    if return_trainer or return_dataset:
        lm_datasets, dataset_length = load_preprocessed_datasets(data_args, model_args)
        train_dataset = lm_datasets["train"]
        val_dataset = lm_datasets["val"]
        val_dataset = val_dataset.select(range(max_val_samples))
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        example = next(iter(train_dataset))
        context_size = len(example["input_ids"])
    else:
        train_dataset = None
        val_dataset = None
        context_size = 6*1024

    segment_size = context_size // (training_args.training_substeps * training_args.segments_per_substep)
    merge_config["segment_size"] = segment_size
    merge_config["context_size"] = context_size

    set_seed(training_args.seed)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    config.summary_length = training_args.summary_length
    config.accumulate_summary = training_args.accumulate_summary
    config.segment_gradient_checkpointing = training_args.segment_gradient_checkpointing
    config.use_kv = training_args.use_kv

    # Create model
    is_llama = (
        "llama" in (model_args.model_name_or_path or model_args.config_name).lower()
    )
    is_deepseek = (
        "deepseek" in (model_args.model_name_or_path or model_args.config_name).lower()
    )
    if is_llama or is_deepseek:
        from auto_compressor import LlamaAutoCompressorModel as AutoCompressorModel
    else:
        from auto_compressor import AutoCompressorModel

    half_dtype = (torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None))
    if model_args.lora or model_args.lora_path or training_args.bf16:
        model_dtype = half_dtype
    else:
        model_dtype = None
    model_kwargs = {"pretrained_model_name_or_path": model_args.model_name_or_path,
        "from_tf":bool(".ckpt" in model_args.model_name_or_path),
        "config":config,
        "cache_dir":model_args.cache_dir,
        "revision":model_args.model_revision,
        "use_auth_token":True if model_args.use_auth_token else None,
        "torch_dtype":model_dtype}
    model = AutoCompressorModel.from_pretrained(**model_kwargs)

    if training_args.fast_attention:
        patch_opt(model)

    tokenizer.padding = True

    if return_trainer:
        trainer = SubstepTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer,
        )
    else:
        trainer = None

    return ModelTrainingModules(trainer, model_kwargs, train_dataset, val_dataset, merge_config)
