import logging
import math
import os
import sys
import torch

import datasets
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    set_seed,
    get_linear_schedule_with_warmup
)

from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from substep_trainer import SubstepTrainer
from base_trainer import EvalCallback
from utils import get_last_checkpoint_or_last_model, parse_checkpoint_step, load_check_merging, wandb_setup
from config_parser import parse_config
import shutil
from pathlib import Path

from data import load_raw_dataset, preprocess_datasets, load_preprocessed_datasets

from fast_attention import patch_opt

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


def save_base_model(config_path, trainer):
    base_model_folder = Path(os.path.join(trainer.args.output_dir, "base_model"))
    Path.mkdir(base_model_folder, exist_ok=True, parents=True)
    shutil.copy2(config_path, os.path.join(base_model_folder, "config_base_model.yaml"))
    trainer.save_model(output_dir=base_model_folder)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config_path = "configs/config.yaml"
    model_args, data_args, training_args, merge_config = parse_config(config_path)
    model_args.use_kv = training_args.use_kv

    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=2000, stdoutToServer=True, stderrToServer=True)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # load_datasets
    if not training_args.do_train:
        data_args.preprocessed_train_datasets = []

    if data_args.preprocessed_train_datasets + data_args.preprocessed_validation_datasets:
        print("train dataset", data_args.preprocessed_train_datasets)
        print("validation dataset", data_args.preprocessed_validation_datasets)

        lm_datasets = load_preprocessed_datasets(data_args, model_args)
    else:
        raw_datasets = load_raw_dataset(data_args, model_args)
        lm_datasets = preprocess_datasets(raw_datasets, tokenizer, data_args, training_args)

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.streaming_data:
            dataset_length = 200000 # TODO replace by real data size
            training_args.max_steps = dataset_length
        else:
            dataset_length = len(train_dataset)
        if data_args.max_train_samples is not None and not data_args.streaming_data:
            max_train_samples = min(dataset_length, data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        print(f"Total number of training data: {dataset_length}")

    example = next(iter(train_dataset))
    context_size = len(example["labels"])
    segment_size = context_size//(training_args.training_substeps*training_args.segments_per_substep)

    if training_args.do_eval:
        # max eval sample deleted
        eval_dataset = {}
        for key in lm_datasets.keys():
            if "validation" in key:
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(data_args.max_eval_samples, len(lm_datasets[key]))
                    eval_dataset[key] = lm_datasets[key].select(range(max_eval_samples))
                else:
                    eval_dataset[key] = lm_datasets[key]

    # Detecting last checkpoint.
    last_checkpoint = None
    if training_args.resume_from_checkpoint:
        if training_args.checkpoint_path is None:
            last_checkpoint = get_last_checkpoint_or_last_model(training_args.output_dir)
        else:
            last_checkpoint = training_args.checkpoint_path
        if last_checkpoint is None:
            print(f"Didn't find a checkpoint in {training_args.output_dir}. Starting training from scratch")
        else:
            print(f"Found checkpoint {last_checkpoint}. Using this checkpoint to resume training.")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

    # Update config with AutoCompressor parameters
    config.summary_length = training_args.summary_length
    config.accumulate_summary = training_args.accumulate_summary
    config.segment_gradient_checkpointing = training_args.segment_gradient_checkpointing
    config.use_kv = training_args.use_kv

    # Create model
    if "llama" in (model_args.model_name_or_path or model_args.config_name).lower():
        from auto_compressor import LlamaAutoCompressorModel as AutoCompressorModel
    else:
        from auto_compressor import AutoCompressorModel

    if model_args.model_name_or_path:
        half_dtype = (torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None))
        if model_args.lora or model_args.lora_path or training_args.bf16:
            model_dtype = half_dtype
        else:
            model_dtype = None
        model = AutoCompressorModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=model_dtype,
        )
    else:
        model = AutoCompressorModel.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # Extend positional embeddings
    if training_args.max_position_embeddings is not None:
        embed = model.model.decoder.embed_positions
        max_pos = model.config.max_position_embeddings
        new_max_pos = training_args.max_position_embeddings
        multiply = math.ceil(new_max_pos / embed.num_embeddings)
        embed.weight.data = torch.cat([
            embed.weight[:-max_pos],
            embed.weight[-max_pos:].repeat(multiply, 1)
        ], dim=0)
        embed.num_embeddings = embed.weight.size(0)
        model.config.max_position_embeddings = max_pos * multiply
        logger.info(f"Positional embeddings increased to {embed.num_embeddings}")

    if training_args.train_embed_only:
        model_args.lora = False
        model_args.lora_path = False

    if model_args.lora or model_args.lora_path:
        from peft import PeftModel, get_peft_model, LoraConfig, TaskType
        if model_args.lora_path:
            logger.info(f"Loading LoRA model from {model_args.lora_path}")
            model = PeftModel.from_pretrained(model, model_args.lora_path)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=model_args.lora_target_modules,
                modules_to_save=model_args.lora_modules_to_save,
            )
            model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    if training_args.fast_attention:
        logger.info("Patching (experimental) fast attention")
        patch_opt(model)

    optimizer = None
    if training_args.train_embed_only:
        num_trainable_parameters = 0
        for name, param in model.named_parameters():
            if 'embed_summary' in name.lower():
                param.requires_grad = True
                num_trainable_parameters += param.numel()
            else:
                param.requires_grad = False
        print(f"Number of trainable parameters = {num_trainable_parameters}")
        optimizer = torch.optim.AdamW(model.embed_summary.parameters(), lr=training_args.learning_rate)
        # num_training_steps = len(train_dataset) * training_args.num_train_epochs
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)
    tokenizer.padding = True

    MyEvalCallback = EvalCallback(batch_size = 5, max_samples = 300, split_size = segment_size)
    trainer = SubstepTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        callbacks=[MyEvalCallback],#GradientLoggerCallback
        optimizers = (optimizer, None)
    )

    # print(trainer.state, trainer.state.is_local_process_zero, accelerator.state.process_index)
    if last_checkpoint is not None:
        if training_args.train_embed_only:
            load_check_merging(last_checkpoint, trainer)
        else:
            trainer._load_from_checkpoint(last_checkpoint)

        process_indx = trainer.accelerator.state.process_index

        if merge_config["resume_run"]:
            wandb_id_file = os.path.join(last_checkpoint, "wandb_id")
            if os.path.exists(wandb_id_file):
                with open(wandb_id_file, "r") as f:
                    run_id = f.read()

                wandb_setup(run_id)
                print(f"resuming Wandb run id = {run_id}")
            else:
                print(f"Could not find run id in ckpt folder. Initializing new run")

        print(f"--- Model loaded in process {process_indx} ---")
    else:
        logger.info("Using a model loaded from scratch!")

    # Training
    if training_args.do_train:
        save_base_model(config_path, trainer)
        trainer.saved_full_model = True
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.saved_full_model = False
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        print(model.state_dict)

        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        if training_args.do_train:
            metrics["global_step"] = trainer.state.global_step
        else:
            if last_checkpoint is None:
                metrics["global_step"] = 0
                last_checkpoint = training_args.output_dir
            else:
                metrics["global_step"] = parse_checkpoint_step(last_checkpoint)
        metrics["model_name"] = last_checkpoint

        if training_args.do_train:
            trainer.log_metrics(f"eval")
            trainer.save_metrics(f"eval")
        else:
            if last_checkpoint is not None:
                step = parse_checkpoint_step(last_checkpoint)
            else:
                step = 0
            segment_string = "-".join([str(i) for i in training_args.segment_lengths])
            metrics["segment_lengths"] = segment_string
            trainer.log_metrics(f"eval_step{step}_{segment_string}", metrics)
            trainer.save_metrics(f"eval_step{step}_{segment_string}", metrics)


if __name__ == "__main__":
    main()
