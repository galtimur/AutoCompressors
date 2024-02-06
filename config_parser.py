import os
from transformers import HfArgumentParser
from args import TrainingArguments, ModelArguments, DataTrainingArguments
from omegaconf import OmegaConf


def parse_config(config_path):
    config = OmegaConf.load(config_path)

    merged_dict = {}
    for d in list(config.values()):
        merged_dict.update(d)
    config = merged_dict

    # Compute additional values
    # TODO pass num_gpus from accelerate config
    config["total_per_device"] = config["total"] // (
        config["num_gpus"] * config["num_nodes"]
    )

    config["config_name"] = config["base_model"]
    config["tokenizer_name"] = config["base_model"]
    config["model_name_or_path"] = config["base_model"]
    config["gradient_accumulation_steps"] = (
        config["total_per_device"] // config["per_device_train_batch_size"]
    )

    # Build the run name suffix based on conditions
    run_name_suffix = f"sub{config['training_substeps']}_seg{config['segments_per_substep']}_sum{config['summary_length']}_lr{config['learning_rate']}_bsz{config['total']}"

    if config["randomize_substeps"]:
        run_name_suffix += "_rand"
    if config["summary_accumulation"]:
        run_name_suffix += "_accu"
    if config["segment_gradient_checkpointing"]:
        run_name_suffix += "_check"

    # run_name=f"{config['base_model']}_config['run_name_suffix']"
    config["run_name"] = f"{config['run_name']}_{run_name_suffix}"
    config["output_dir"] = os.path.join(config["out_dir"], config["run_name"])
    os.makedirs(config["output_dir"], exist_ok=True)

    # Set environment variables
    os.environ["WANDB_DIR"] = config["out_dir"]
    os.environ["WANDB_PROJECT"] = "autocompressors"
    os.environ["OMP_NUM_THREADS"] = "8"

    keys_to_remove = [
        "base_model",
        "checkpointing",
        "dir",
        "eval_domains",
        "node",
        "num_gpus",
        "num_nodes",
        "out_dir",
        "project",
        "summary_accumulation",
        "total",
        "total_per_device",
        "train_domains",
    ]

    config = {k: v for k, v in config.items() if k not in keys_to_remove}
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args_dict, data_args_dict, training_args_dict = parser.parse_dict(config)

    return model_args_dict, data_args_dict, training_args_dict
