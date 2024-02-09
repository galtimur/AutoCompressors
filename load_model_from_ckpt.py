from auto_compressor import LlamaAutoCompressorModel as AutoCompressorModel
from transformers import LlamaConfig

import os
from omegaconf import OmegaConf

from utils import merge_ckpts

# TODO check that llama is the model
# if "llama" in (model_args.model_name_or_path or model_args.config_name).lower():


def load_model_from_ckpt(checkpoint_path):

    base_folder = os.path.dirname(checkpoint_path)
    temp_folder = os.path.join(base_folder, "checkpoint_merge_temp")
    main_folder = os.path.join(base_folder, "base_model")
    merge_ckpts(main_folder, checkpoint_path, temp_folder, config_filename="config_base_model.yaml")
    config_path = os.path.join(main_folder, "config_base_model.yaml")

    config = LlamaConfig.from_pretrained(temp_folder)

    model = AutoCompressorModel.from_pretrained(
        temp_folder,
        config=config,
        torch_dtype=config.torch_dtype,
    )

    run_config = OmegaConf.load(config_path)
    run_config = OmegaConf.to_container(run_config, resolve=True)

    return model, run_config

checkpoint_path = "/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_embed_only_test/checkpoint-9900"

model, run_config = load_model_from_ckpt(checkpoint_path)

print(1)