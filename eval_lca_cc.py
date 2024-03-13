from torch.utils.data import Dataset
import torch
import json
import time
import re
from datasets import load_dataset, load_from_disk
from pathlib import Path
from load_model_from_ckpt import load_model_from_ckpt, load_base_model
from utils import get_torch_device
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
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
    checkpoint_path: str | Path,
    ds_test: str | None,
    dataset_ppl: str | None,
    model_name: str,
    results_directory: str | Path,
    results_filename: str | Path,
    do_dump_results: bool = True,
    gpu_num: int | None = None,
    limit: int | None = None,
) -> dict:

    device = "cuda:0"
    if model_name.startswith("base_model"):
        model, tokenizer = load_base_model("deepseek-ai/deepseek-coder-1.3b-base")
        match = re.search(r'\d+$', model_name)
        context_size = int(match.group())
    else:
        model, tokenizer, run_config = load_model_from_ckpt(checkpoint_path)
        context_size = 6 * 1024
    model.eval()
    model.to(device)
    device = get_torch_device(gpu_num)
    stopping_criteria = StoppingCriteriaList([StopOnNewLine(tokenizer)])
    if ds_test is None:
        ds_test = LcaPythonCompletionDataset()
    max_comp = 128
    max_len_model = context_size  # model.config.max_position_embeddings
    max_len_ctx = max_len_model - max_comp

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
        # prompt = tokenizer.decode(input_ids[0])
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_comp,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
        out_tokens = out[0, len(input_ids[0]) - 1 :]
        pred = tokenizer.decode(out_tokens).strip("\n")
        preds.append(pred)
        grtrs.append(sample["gt"])
        if pred != sample["gt"]:
            with open(f'out/false_preds_{model_name}.txt', 'a') as f:
                f.write(f"{pred} --> {sample['gt']}\n")

    time_used_lca = time.time() - start_time
    exact_match = sum(gt == pr for gt, pr in zip(grtrs, preds)) / len(preds)

    start_time = time.time()
    max_loss_samples = 1000
    if dataset_ppl is not None:

        if not model_name.startswith("base_model"):
            batch_size = 8
            segment_size = context_size // (
                    run_config["training_substeps"] * run_config["segments_per_substep"]
            )
            eval_result = evaluate_ppl_red_pajamas(
                model,
                dataset_ppl,
                batch_size,
                max_samples=max_loss_samples,
                split_size=segment_size,
                disable_tqdm=False,
            )
            av_loss = eval_result["total_loss"]
        else:
            av_loss = evaluate_base_model(model, dataset_ppl, batch_size=1, max_samples=max_loss_samples, context_size=context_size)
    time_used_loss = time.time() - start_time
    results = {
        "task": "lca_completion",
        'model': model_name,
        "exact_match_rate": exact_match,
        "loss": av_loss,
        "LCA items/s": num_samples/time_used_lca,
        "loss items/s": max_loss_samples/time_used_loss,
        "number of LCA items": num_samples,
        "number of loss items": max_loss_samples,
        "checkpoint_path": checkpoint_path,
    }
    if do_dump_results:
        dump_results(results, results_directory, results_filename)
    return results

def eval_models_on_lcc(ckpt_map_path: str | Path, results_path: str | Path, limit: int | None = None):
    with open(ckpt_map_path, 'r') as f:
        ckpt_name_map = json.load(f)
    dataset = LcaPythonCompletionDataset()
    dataset_ppl = load_from_disk("/mnt/data2/shared-data/autocompressors/6k_py_320000_samp/valid")
    dataset_ppl = dataset_ppl.shuffle(seed=42)
    for model_name, ckpt_path in ckpt_name_map.items():
        print(f"Running {model_name}")
        eval_result = eval_on_lcc(
            ckpt_path,
            dataset,
            dataset_ppl=dataset_ppl,
            model_name=model_name,
            results_directory="out",
            results_filename="eval_lca_cc.json",
            do_dump_results=False,
            limit=limit,
        )
        with open(results_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(eval_result) + "\n")


if __name__ == "__main__":
    ckpt_map_path = 'configs/ckpt_name_map.json'
    results_path = "out/eval_lca_cc.json"
    eval_models_on_lcc(ckpt_map_path, results_path, limit = 2000)
