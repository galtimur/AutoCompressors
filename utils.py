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

def check_proc_flags(folder: str, max_proc: int, prefix: str):
    files = os.listdir(folder)
    expected_files = [f'{prefix}_{i}' for i in range(max_proc)]
    all_files_exist = all(file in files for file in expected_files)

    return all_files_exist

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

def wandb_setup(run_id):

    os.environ["WANDB_RESUME"] = "must"
    os.environ["WANDB_RUN_ID"] = run_id

def traferse_folder(folder: str) -> set:
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
