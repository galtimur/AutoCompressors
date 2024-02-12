from pathlib import Path

from huggingface_hub.utils import HFValidationError
from omegaconf import OmegaConf
from peft import PeftModel, get_peft_model, LoraConfig
from tokenizers import Tokenizer
from transformers import LlamaConfig, AutoTokenizer

from auto_compressor import LlamaAutoCompressorModel
from utils import merge_ckpts


# TODO check that llama is the model
# if "llama" in (model_args.model_name_or_path or model_args.config_name).lower():


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


def load_lora_model_from_ckpt(checkpoint_path: str, merged_config: dict):
    checkpoint_path = Path(checkpoint_path)
    base_folder = checkpoint_path.parent
    main_folder = base_folder / "base_model"
    config = LlamaConfig.from_pretrained(main_folder)

    model = LlamaAutoCompressorModel.from_pretrained(main_folder, config=config, torch_dtype=config.torch_dtype)

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
                         ) -> tuple[LlamaAutoCompressorModel, Tokenizer, dict]:

    checkpoint_path = Path(checkpoint_path)
    base_folder = checkpoint_path.parent
    main_folder = base_folder / "base_model"

    config, merged_config = load_flat_config(main_folder / "config_base_model.yaml")

    if merged_config["lora"]:
        model, tokenizer = load_lora_model_from_ckpt(checkpoint_path, merged_config)
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
