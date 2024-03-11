import argparse
import time
import json
import gc
from omegaconf import OmegaConf
from tqdm import tqdm

from auto_compressor import LlamaAutoCompressorModel as AutoCompressorModel

import torch
from torch.utils.data import DataLoader

from utils import get_run_mudules
from eval_ppl import evaluate_ppl_red_pajamas


def collate_fn(batch):
    cll_batch = dict()
    for k in batch[0].keys():
        cll_batch[k] = torch.stack([torch.tensor(s[k], device="cuda:0") for s in batch])
    return cll_batch


def speed_bench(args, config_path, result_file):
    bench_config = OmegaConf.load("configs/benchmark_params_backup.yaml")
    bench_config = OmegaConf.to_container(bench_config, resolve=True)

    eval_bz_lst = bench_config["per_device_eval_batch_size"]
    train_bz_lst = bench_config["per_device_train_batch_size"]
    compress_rate = bench_config["summary_сompression"]
    segments_per_substep_lst = bench_config["segments_per_substep"]
    training_substeps_lst = bench_config["training_substeps"]
    max_train_samples = bench_config["max_train_samples"]
    max_val_samples = bench_config["max_val_samples"]

    for (
        segments_per_substep,
        training_substeps,
        train_bz,
        eval_bz,
    ) in zip(
        segments_per_substep_lst, training_substeps_lst, train_bz_lst, eval_bz_lst
    ):
        segments_number = segments_per_substep * training_substeps
        summary_length = 6000 // (segments_number * compress_rate)
        if segments_number == 1:
            summary_length = 0

        # if segments_number < 10:
        #     continue

        bench_parameters = {
            "segments_per_substep": segments_per_substep,
            "training_substeps": training_substeps,
            "per_device_train_batch_size": train_bz,
            "per_device_eval_batch_size": eval_bz,
            "summary_length": summary_length,
            "max_train_samples": max_train_samples,
        }
        # if training_substeps != 8:
        #     continue
        ModelTrainingModules = get_run_mudules(
            args, config_path, bench_parameters, max_val_samples
        )

        trainer, model_kwargs, train_dataset, val_dataset, merge_config = (
            ModelTrainingModules.trainer,
            ModelTrainingModules.model_kwargs,
            ModelTrainingModules.train_dataset,
            ModelTrainingModules.val_dataset,
            ModelTrainingModules.merge_config,
        )

        print(bench_parameters)

        context_size = merge_config["context_size"]
        segment_size = merge_config["segment_size"]
        eval_batch = merge_config["per_device_eval_batch_size"]

        bench_result = dict()

        tokens_to_generate = segment_size // 2 if segments_number > 1 else 512
        bench_result["train batch size"] = segments_number
        bench_result["eval batch size"] = segments_number
        bench_result["segments_number"] = segments_number
        bench_result["segments_number"] = segments_number
        bench_result["segment_size"] = segment_size
        bench_result["segments_per_substep"] = segments_per_substep
        bench_result["training_substeps"] = training_substeps
        bench_result["tokens forward"] = len(dataset) * context_size
        bench_result["tokens eval"] = len(val_dataset) * context_size
        bench_result["tokens generated"] = len(dataset) * tokens_to_generate
        bench_result["tokens prompt gen"] = len(dataset) * (
            context_size - segment_size // 2
        )

        # Full model does not fit single 24GB GPU
        if segments_number > 1:
            print("-------------- performing train -------------")
            start_time = time.time()
            trainer.train()
            bench_result["train speed tok/s"] = bench_result["tokens forward"] / (
                time.time() - start_time
            )

        device = trainer.model.device
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoCompressorModel.from_pretrained(**model_kwargs)
        model.to(device)

        print("-------------- performing eval -------------")
        model.eval()
        start_time = time.time()
        evaluate_ppl_red_pajamas(
            model,
            val_dataset,
            eval_batch,
            max_samples=-1,
            split_size=segment_size,
            disable_tqdm=False,
        )
        bench_result["eval speed tok/s"] = bench_result["tokens eval"] / (
            time.time() - start_time
        )

        print("------------ performing generation -----------")
        model.eval()
        model.gen_mode = True
        data_loader = DataLoader(dataset, eval_batch, collate_fn=collate_fn)

        start_time = time.time()
        for item in tqdm(data_loader):
            input_ids = item["input_ids"][:, : -segment_size // 2]

            output_ids = model.generate(
                input_ids,
                max_length=(segment_size // 2) + tokens_to_generate,
                segment_lengths=segment_size,
                do_sample=False,
            )
            print(output_ids.size(1) - input_ids.size(1))
        bench_result["generation speed tok/s"] = bench_result["tokens generated"] / (
            time.time() - start_time
        )

        with open(result_file, "a") as jsonl_file:
            jsonl_file.write(json.dumps(bench_result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', help='Path to the config file')
    parser.add_argument("--suffix", help="Any suffix for run name")
    parser.add_argument(
        "--dev", action="store_true", help='Dev mode, adds "test" to the prefix'
    )
    args = parser.parse_args()
    config_path = "configs/config_speeed_benchmark.yaml"
    result_file = "out/time_bench_result.jsonl"
    speed_bench(args, config_path, result_file)
