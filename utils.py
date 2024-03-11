import os
import re
import time
import configparser
from dataclasses import dataclass

import shutil

import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)

from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from substep_trainer import SubstepTrainer
from config_parser import parse_config

from data import load_preprocessed_datasets
from fast_attention import patch_opt

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def get_aws_credentials_local(cred_file: str = "~/.aws/credentials"):
    aws_credentials_path = os.path.expanduser(cred_file)

    if os.path.exists(aws_credentials_path):
        config = configparser.ConfigParser()
        config.read(aws_credentials_path)

        if 'default' in config:
            access_key_id = config['default'].get('aws_access_key_id')
            secret_access_key = config['default'].get('aws_secret_access_key')

            if access_key_id and secret_access_key:
                print("Found AWS creds locally")
                return access_key_id, secret_access_key

    print(f"Could not find proper file {cred_file}")

    return None, None


def get_last_checkpoint_or_last_model(folder):
    """modification of get_last_checkpoint from transformer.trainer_utils.
    This function will return the main folder if it contains files of the form "pytorch_model*". The default HF function ignores those and only looks
    for "checkpoint-*" folders."""
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    _re_model = re.compile("pytorch_model" + r"*")
    content = os.listdir(folder)
    models = [
        path for path in content if _re_model.search(path) is not None
    ]
    if models != []:
        return folder
    else:
        checkpoints = [
            path
            for path in content
            if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
        ]
        if len(checkpoints) == 0:
            return
        return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def parse_checkpoint_step(checkpoint):
    if checkpoint.split("-")[0]!= "checkpoint":
        return -1
    else:
        try:
            return int(checkpoint.split("-")[-1])
        except:
            print(f"got checkpoint name {checkpoint}, couldn't parse step")
            return -1

def calc_grad(model):
    total_norm = 0.0
    num_parameters = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
            num_parameters += p.numel()
    total_norm = (total_norm**0.5) / (num_parameters+1)

    return total_norm

def check_proc_flags(folder: str, max_proc: int, prefix: str):
    files = os.listdir(folder)
    expected_files = [f'{prefix}_{i}' for i in range(max_proc)]
    all_files_exist = all(file in files for file in expected_files)

    return all_files_exist

def load_check_merging(last_checkpoint: str, trainer):
    process_indx = trainer.accelerator.state.process_index
    max_proc = trainer.accelerator.num_processes
    base_folder = os.path.dirname(last_checkpoint)
    temp_folder = os.path.join(base_folder, "checkpoint_merge_temp")
    flag_filename = ".merging_done_flag"
    flag_file = os.path.join(temp_folder, flag_filename)
    flag_prefix = ".flag_proc"
    # TODO add node index too
    flag_file_process = os.path.join(temp_folder, f"{flag_prefix}_{process_indx}")
    if trainer.state.is_local_process_zero and trainer.state.is_world_process_zero:
        main_model_folder = os.path.join(base_folder, "base_model")
        config_filename = "config_base_model.yaml"
        merge_ckpts(main_model_folder, last_checkpoint, temp_folder, flag_filename, config_filename)
    else:
        exist_merge = os.path.exists(flag_file)
        while not exist_merge:
            exist_merge = os.path.exists(flag_file)
            time.sleep(0.2)

    trainer._load_from_checkpoint(temp_folder)
    with open(flag_file_process, 'w') as f:
        pass

    wait = not check_proc_flags(temp_folder, max_proc, flag_prefix)
    while wait:
        wait = not check_proc_flags(temp_folder, max_proc, flag_prefix)
        time.sleep(0.2)
    if trainer.state.is_local_process_zero and trainer.state.is_world_process_zero:
        shutil.rmtree(temp_folder)

def wandb_setup(run_id):

    os.environ["WANDB_RESUME"] = "must"
    os.environ["WANDB_RUN_ID"] = run_id

def traverse_folder(folder: str) -> set:
    file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return set(file_paths)

def save_set(set_to_save: set, file: str):
    with open(file, 'w') as f:
        for item in set_to_save:
            f.write(item + "\n")

def load_set(file: str):
    with open(file, 'r') as f:
        return set([line.strip() for line in f])

def get_torch_device(gpu_num: int | None) -> torch.device:
    if gpu_num is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        print(gpu_num)
        device = torch.device(f'cuda:{gpu_num}')
    return device

@dataclass
class ModelTrainingModules:
    trainer: object
    model_kwargs: dict
    train_dataset: object
    val_dataset: object
    merge_config: object

def get_run_mudules(args, config_path, bench_parameters, max_val_samples, return_dataset=True, return_trainer=True):

    model_args, data_args, training_args, merge_config = parse_config(config_path, args, override_dict=bench_parameters)
    model_args.use_kv = training_args.use_kv

    if return_trainer or return_dataset:
        lm_datasets, dataset_length = load_preprocessed_datasets(data_args, model_args)
        train_dataset = lm_datasets["train"]
        val_dataset = train_dataset.select(range(max_val_samples))
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
        example = next(iter(train_dataset))
        context_size = len(example["labels"])
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
    if "llama" in (model_args.model_name_or_path or model_args.config_name).lower():
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
