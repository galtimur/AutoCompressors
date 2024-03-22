from torch.utils.data import Dataset
import torch
import json
import time
import re
from datasets import load_dataset, load_from_disk
from pathlib import Path
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from load_model_from_ckpt import load_model_from_ckpt, load_base_model
from eval_ppl import evaluate_ppl_red_pajamas, evaluate_base_model


def dump_results(
    results: dict, results_dir: str | Path, results_filename: str | Path
) -> None:
    results_dir = Path(results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / results_filename, "w") as f:
        print("Dumping to", str(results_dir / results_filename))
        json.dump(results, f)


class LcaPythonCompletionDataset(Dataset):
    dataset_name = "jenyag/repo-codegen-py-py-context-path-distance"

    def __init__(self) -> None:
        dataset = load_dataset(self.dataset_name)["test"]
        self.samples = []
        for sample in dataset:
            for context, ground_truth in zip(sample["file_context"], sample["gt"]):
                context = sample["project_context"] + context["content"]
                if len(context) == 0:
                    continue
                if context[-1] != "\n":
                    context = context + "\n"
                self.samples.append({"context": context, "gt": ground_truth})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, str]:
        return self.samples[idx]


class StopOnNewLine(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = set()
        for k, tok_id in tokenizer.get_vocab().items():
            s = tokenizer.convert_tokens_to_string([k])
            if "\n" in s:
                self.stop_ids.add(tok_id)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert input_ids.shape[0] == 1  # only batch_size 1 is supported
        if input_ids[0, -1].item() in self.stop_ids:
            return True
        else:
            return False

def eval_on_lcc(
        modules: dict,
        ds_test: str | None,
        context_size: int,
        limit: int | None = None,
) -> dict:

    model, tokenizer, run_config, model_name = modules["model"], modules["tokenizer"], modules["run_config"], modules["model_name"]
    device = model.device

    stopping_criteria = StoppingCriteriaList([StopOnNewLine(tokenizer)])
    if ds_test is None:
        ds_test = LcaPythonCompletionDataset()
    max_comp = 128  # max length of the line
    # context_size = model.config.max_position_embeddings
    max_len_ctx = context_size - max_comp  # input context should be less that model context size minus max line length
    assert max_len_ctx > 0, "max_len_ctx should be positive!"

    grtrs = []
    preds = []

    num_samples = len(ds_test) if limit is None else limit
    ds_test = ds_test[:num_samples]
    with open(f'out/false_preds_{model_name}.txt', 'a') as f:
        f.write("----- New eval -----\n")

    start_time = time.time()
    for sample in tqdm(ds_test):
        input_ids = tokenizer.encode(sample["context"], return_tensors="pt")
        input_ids = input_ids[:, -max_len_ctx:].to(device)

        model_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_comp,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": False,
        }
        if not model_name.startswith("base_model"):
            model_kwargs["segment_lengths"] = 1024

        with torch.no_grad():
            out = model.generate(**model_kwargs)

        out_tokens = out[0, len(input_ids[0]) - 1:]
        pred = tokenizer.decode(out_tokens).strip("\n")
        preds.append(pred)
        grtrs.append(sample["gt"])
        if pred != sample["gt"]:
            with open(f'out/false_preds_{model_name}.txt', 'a') as f:
                f.write(f"{pred} --> {sample['gt']}\n")

    time_used_lca = time.time() - start_time
    exact_match = sum(gt == pr for gt, pr in zip(grtrs, preds)) / len(preds)

    results = {
        "exact_match_rate": exact_match,
        "LCA items/s": num_samples / time_used_lca,
        "number of LCA items": num_samples,
    }

    return results


def eval_cross_entropy(
    modules: dict,
    dataset_ce: str | None,
    context_size: int,
    limit_loss_samples: int | None = None,
) -> dict:

    model, tokenizer, run_config, model_name = modules["model"], modules["tokenizer"], modules["run_config"], modules["model_name"]

    start_time = time.time()
    if dataset_ce is not None:

        if not model_name.startswith("base_model") or model_name.startswith("fintuned"):
            batch_size = 8
            num_segments = run_config["training_substeps"] * run_config["segments_per_substep"]
            segment_length = 6 * 1024 // num_segments

            eval_result = evaluate_ppl_red_pajamas(
                model,
                dataset_ce,
                batch_size,
                max_samples=limit_loss_samples,
                split_size=segment_length,
                disable_tqdm=False,
            )
            av_loss = eval_result["total_loss"]
        else:
            av_loss = evaluate_base_model(model, dataset_ce, batch_size=1, max_samples=limit_loss_samples, context_size=context_size)
    time_used_loss = time.time() - start_time
    results = {
        "loss": av_loss,
        "loss items/s": limit_loss_samples/time_used_loss,
        "number of loss items": limit_loss_samples,
    }

    return results


def eval_model(
    checkpoint_path: str | Path,
    ds_test: str | None,
    dataset_ce: str | None,
    model_name: str,
    limit: int | None = None,
    limit_loss_samples: int | None = None,
    do_lcc: bool = True,
    do_ce_loss: bool = True,
    ) -> dict:

    modules = {}
    device = "cuda:0"
    if model_name.startswith("base_model"):
        model, modules["tokenizer"] = load_base_model("deepseek-ai/deepseek-coder-1.3b-base")
        modules["run_config"] = None
    else:
        model, modules["tokenizer"], modules["run_config"] = load_model_from_ckpt(checkpoint_path)

    modules["model_name"] = model_name

    match = re.search(r'_([0-9]+)$', model_name)
    if match is not None:
        context_size = int(match.group()[1:])
    else:
        context_size = 6 * 1024
    model.eval()
    model.to(device)
    modules["model"] = model

    results = {'model': model_name}

    if do_lcc:
        result_lcc = eval_on_lcc(
            modules,
            ds_test,
            context_size,
            limit)
        results.update(result_lcc)

    if do_ce_loss:
        result_ce = eval_cross_entropy(
            modules,
            dataset_ce,
            context_size,
            limit_loss_samples)
        results.update(result_ce)

    results.update({"checkpoint_path": checkpoint_path})

    return results

def run_benchmark(
        ckpt_map_path: str | Path,
        results_path: str | Path,
        limit: int | None = None,
        limit_loss_samples: int | None = None,
        do_lcc: bool = True,
        do_ce_loss: bool = True):
    with open(ckpt_map_path, 'r') as f:
        ckpt_name_map = json.load(f)
    dataset_lcc = LcaPythonCompletionDataset()
    dataset_ce = load_from_disk("/mnt/data2/shared-data/autocompressors/6k_py_320000_samp/valid")
    dataset_ce = dataset_ce.shuffle(seed=42)
    for model_name, ckpt_path in ckpt_name_map.items():
        print(f"Running {model_name}")
        eval_result = eval_model(
            ckpt_path,
            dataset_lcc,
            dataset_ce=dataset_ce,
            model_name=model_name,
            limit=limit,
            limit_loss_samples=limit_loss_samples,
            do_lcc = do_lcc,
            do_ce_loss = do_ce_loss,
        )
        with open(results_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(eval_result) + "\n")


if __name__ == "__main__":
    ckpt_map_path = 'configs/ckpt_name_map.json'
    results_path = "out/eval_lca_cc.json"
    run_benchmark(ckpt_map_path, results_path, limit = 20, limit_loss_samples = 20, do_lcc=False, do_ce_loss=True)
