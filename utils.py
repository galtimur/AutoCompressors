import os
import re
import time

from safetensors import safe_open
from safetensors.torch import save_file
import shutil

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

def load_check_merging(last_checkpoint: str, trainer):
    base_folder = os.path.dirname(last_checkpoint)
    flag_file = os.path.join(base_folder, "merging_done.flag")
    if trainer.state.is_local_process_zero and trainer.state.is_world_process_zero:
        file_path_main = os.path.join(base_folder, "model.safetensors")
        file_path_part = os.path.join(last_checkpoint, "model.safetensors")
        file_path_part_copy = os.path.join(last_checkpoint, "model_copy.safetensors_copy")
        shutil.copy2(file_path_part, file_path_part_copy)
        tensors = {}
        with safe_open(file_path_main, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        with safe_open(file_path_part, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        save_file(tensors, file_path_part)

        with open(flag_file, 'w') as f:
            pass

    else:
        exist_merge = os.path.exists(flag_file)
        while exist_merge:
            exist_merge = os.path.exists(flag_file)
            time.sleep(0.5)

    trainer._load_from_checkpoint(last_checkpoint)

    if trainer.state.is_local_process_zero and trainer.state.is_world_process_zero:
        print("-------- setting back -------")
        time.sleep(5)
        shutil.copy2(file_path_part_copy, file_path_part)
        os.remove(file_path_part_copy)
        os.remove(flag_file)