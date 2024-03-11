from torch.utils.data import Dataset
import torch
import json
from datasets import load_dataset
from pathlib import Path
from load_model_from_ckpt import load_model_from_ckpt
from utils import get_torch_device
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm


def dump_results(
    results: dict, results_dir: str | Path, results_filename: str | Path
) -> None:
    results_dir = Path(results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / results_filename, "w") as fp:
        print("Dumping to", str(results_dir / results_filename))
        json.dump(results, fp)


class LcaPythonCompletionDataset(Dataset):
    dataset_name = "jenyag/repo-codegen-py-py-context-path-distance"

    def __init__(self) -> None:
        ds = load_dataset(self.dataset_name)["test"]
        self.samples = []
        for s in ds:
            for context, gt in zip(s["file_context"], s["gt"]):
                context = s["project_context"] + context["content"]
                if len(context) == 0:
                    continue
                if context[-1] != "\n":
                    context = context + "\n"
                self.samples.append({"context": context, "gt": gt})

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
    model_name: str,
    results_directory: str | Path,
    results_filename: str | Path,
    do_dump_results: bool = True,
    gpu_num: int | None = None,
    limit: int | None = None,
) -> dict:
    device = "cuda:0"
    model, tokenizer, run_config = load_model_from_ckpt(checkpoint_path)
    model.eval()
    model.to(device)
    ds_test = LcaPythonCompletionDataset()
    device = get_torch_device(gpu_num)
    stopping_criteria = StoppingCriteriaList([StopOnNewLine(tokenizer)])

    max_comp = 128
    max_len_model = 6 * 1024  # model.config.max_position_embeddings
    max_len_ctx = max_len_model - max_comp

    grtrs = []
    preds = []

    num_samples = len(ds_test) if limit is None else limit
    ds_test = ds_test[:limit]

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

    exact_match = sum(gt == pr for gt, pr in zip(grtrs, preds)) / len(preds)
    results = {
        "task": "lcc",
        'model': model_name,
        "exact_match_rate": exact_match,
        "checkpoint_path": checkpoint_path,
    }
    if do_dump_results:
        dump_results(results, results_directory, results_filename)
    return results


if __name__ == "__main__":
    # ckpt_path = "/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_new_split/checkpoint-17100"
    ckpt_path = "/mnt/data2/arkhipov/experiments/autocompressors/deepseek-1.3B_sub3_seg2_sum50_code_base/checkpoint-10000"
    # with open('path_to_file/person.json', 'r') as f:
    #     data = json.load(f)
    eval_on_lcc(
        ckpt_path,
        model_name="model_name",
        results_directory="out",
        results_filename="eval_lca_cc.json",
        do_dump_results=True,
        limit=5,
    )
