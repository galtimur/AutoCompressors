from pathlib import Path
import os
import shutil
import time

from safetensors import safe_open
from safetensors.torch import save_file

from huggingface_hub.utils import HFValidationError
from omegaconf import OmegaConf
from peft import PeftModel, get_peft_model, LoraConfig
from tokenizers import Tokenizer
from transformers import LlamaConfig, AutoTokenizer

from auto_compressor import LlamaAutoCompressorModel
from utils import check_proc_flags


# TODO check that llama is the model
# if "llama" in (model_args.model_name_or_path or model_args.config_name).lower():

def merge_ckpts(main_folder, part_folder, temp_folder, flag_filename=".merging_done_flag", config_filename = "config_base_model.yaml"):
    flag_file = os.path.join(temp_folder, flag_filename)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    shutil.copytree(part_folder, temp_folder)
    shutil.copy2(os.path.join(main_folder, config_filename), os.path.join(temp_folder, config_filename))
    file_path_part = os.path.join(part_folder, "model.safetensors")
    file_path_main = os.path.join(main_folder, "model.safetensors")
    model_tensor_path = os.path.join(temp_folder, "model.safetensors")
    os.remove(model_tensor_path)
    tensors = {}
    with safe_open(file_path_main, framework="pt") as f:
        metadata = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    with safe_open(file_path_part, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    save_file(tensors, model_tensor_path, metadata)
    with open(flag_file, 'w') as f:
        pass

    return flag_file

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


def load_flat_config(config_path):

    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {}
    for d in list(config.values()):
        merged_config.update(d)

    return config, merged_config


def load_only_embed_model_from_ckpt(checkpoint_path: str, merged_config):

    checkpoint_path = Path(checkpoint_path)
    base_folder = checkpoint_path.parent
    main_folder = base_folder / "base_model"
    temp_folder = base_folder / "checkpoint_merge_temp"

    # TODO refactor merge_ckpts to use Path objects
    merge_ckpts(str(main_folder), str(checkpoint_path), str(temp_folder), config_filename="config_base_model.yaml")

    config = LlamaConfig.from_pretrained(temp_folder)
    model = LlamaAutoCompressorModel.from_pretrained(
        temp_folder,
        config=config,
        torch_dtype=config.torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(temp_folder, use_fast=merged_config["use_fast_tokenizer"])

    return model, tokenizer


def load_lora_model_from_ckpt(checkpoint_path: str | Path,
                              base_model_dir: str | Path,
                              merged_config: dict) -> tuple[LlamaAutoCompressorModel, Tokenizer]:
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    config = LlamaConfig.from_pretrained(base_model_dir)

    model = LlamaAutoCompressorModel.from_pretrained(checkpoint_path, config=config, torch_dtype=config.torch_dtype)

    try:
        model = PeftModel.from_pretrained(model, checkpoint_path)
    except HFValidationError as e:
        raise e
        print('Tried to load PEFT model but failed. Trying to load as a model+peft checkpoint')
        with open(checkpoint_path / 'adapter_config.json') as fp:
            peft_config_dict = json.load(fp)
        print(json.dumps(peft_config_dict, indent=4))
        peft_config = LoraConfig(peft_config_dict)
        model = get_peft_model(model, peft_config)
        model.load_state_dict(torch.load(checkpoint_path / 'pytorch_model.bin', map_location='cpu'))

    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=merged_config["use_fast_tokenizer"])

    return model, tokenizer


def load_model_from_ckpt(checkpoint_path: str | Path,
                         base_model_dir: str | Path | None = None
                         ) -> tuple[LlamaAutoCompressorModel, Tokenizer, dict]:

    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    if base_model_dir is None:
        base_folder = checkpoint_path.parent
        main_folder = base_folder / "base_model"
    else:
        main_folder = Path(base_model_dir)

    config, merged_config = load_flat_config(main_folder / "config_base_model.yaml")

    if merged_config["lora"]:
        print(checkpoint_path)
        model, tokenizer = load_lora_model_from_ckpt(checkpoint_path, main_folder, merged_config)
        # TODO If you want, I can add function that saves the fused model as a checkpoint
        print("Loaded LORA model")
    elif merged_config["train_embed_only"]:
        model, tokenizer = load_only_embed_model_from_ckpt(checkpoint_path, merged_config)
        print("Loaded embeddings only model")
    else:
        print("Loading for this config state is not implemented yet")

    return model, tokenizer, merged_config


if __name__ == '__main__':
    # checkpoint_path = "/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_embed_only_test/checkpoint-9900"
    ckpt_path = "/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50/checkpoint-2250"
    model, tokenizer, run_config = load_model_from_ckpt(ckpt_path)
    print(1)
    pass