from torch.utils.data import Dataset
import torch
from datasets import load_dataset
from pathlib import Path
from load_model_from_ckpt import load_model_from_ckpt
from utils import get_torch_device
from transformers.generation import StoppingCriteria, StoppingCriteriaList


class LcaPythonCompletionDataset(Dataset):
    dataset_name = 'jenyag/repo-codegen-py-py-context-path-distance'

    def __init__(self) -> None:
        ds = load_dataset(self.dataset_name)['test']
        self.samples = []
        for s in ds:
            for context, gt in zip(s['file_context'], s['gt']):
                context = s['project_context'] + context['content']
                if len(context) == 0:
                    continue
                if context[-1] != '\n':
                    context = context + '\n'
                self.samples.append({'context': context, 'gt': gt})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, str]:
        return self.samples[idx]

class StopOnNewLine(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = set()
        for k, tok_id in tokenizer.vocab.items():
            s = tokenizer.convert_tokens_to_string([k])
            if '\n' in s:
                self.stop_ids.add(tok_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.shape[0] == 1  # only batch_size 1 is supported
        if input_ids[0, -1].item() in self.stop_ids:
            return True
        else:
            return False

def eval_on_lcc(checkpoint_path: str | Path,
                results_directory: str | Path,
                results_filename: str | Path,
                gpu_num: int | None,
                limit: int | None) -> dict:

    device = "cuda:0"
    model, tokenizer, run_config = load_model_from_ckpt(checkpoint_path)
    model.eval()
    model.to(device)
    ds_test = LcaPythonCompletionDataset()
    device = get_torch_device(gpu_num)
    stopping_criteria = StoppingCriteriaList([StopOnNewLine(tokenizer)])

    # max_len_model = model.config.max_position_embeddings
    max_comp = 128
    # max_len_ctx = max_len_model - max_comp
    #
    # grtrs = []
    # preds = []
    #
    num_samples = len(ds_test) if limit is None else limit
    # for n in range(num_samples):
    #     s = ds_test[n]
    #     input_ids = tokenizer.encode(s['context'], return_tensors='pt')
    #     input_ids = input_ids[:, -max_len_ctx:].to(device)
    #     with torch.no_grad():
    #         out = model.generate(input_ids, max_new_tokens=max_comp, stopping_criteria=stopping_criteria)
    #     out_tokens = out[0, len(input_ids[0]) - 1:]
    #     pred = tokenizer.decode(out_tokens).strip('\n')
    #     preds.append(pred)
    #     grtrs.append(s['gt'])
    #
    # em = sum(gt == pr for gt, pr in zip(grtrs, preds)) / len(preds)
    # results = {
    #     'task': 'lcc',
    #     'model': model_name,
    #     'em': em
    # }
    # dump_results(results, results_directory, results_filename)
    return None

if __name__ == "__main__":
    ckpt_path = "/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50/checkpoint-2250"
    eval_on_lcc(checkpoint_path, results_directory = "out", results_filename="out/eval_lca_cc.json", limit = 10)