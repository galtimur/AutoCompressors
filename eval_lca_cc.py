from torch.utils.data import Dataset
from datasets import load_dataset
from pathlib import Path


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


def eval_on_lcc(model_name: str | Path,
                results_directory: str | Path,
                results_filename: str | Path,
                gpu_num: int | None,
                limit: int | None) -> dict:

    ds_test = LcaPythonCompletionDataset()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = get_torch_device(gpu_num)
    stopping_criteria = StoppingCriteriaList([StopOnNewLine(tokenizer)])
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    max_len_model = model.config.max_position_embeddings
    max_comp = 128
    max_len_ctx = max_len_model - max_comp

    grtrs = []
    preds = []

    num_samples = len(ds_test) if limit is None else limit
    for n in range(num_samples):
        s = ds_test[n]
        input_ids = tokenizer.encode(s['context'], return_tensors='pt')
        input_ids = input_ids[:, -max_len_ctx:].to(device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_comp, stopping_criteria=stopping_criteria)
        out_tokens = out[0, len(input_ids[0]) - 1:]
        pred = tokenizer.decode(out_tokens).strip('\n')
        preds.append(pred)
        grtrs.append(s['gt'])

    em = sum(gt == pr for gt, pr in zip(grtrs, preds)) / len(preds)
    results = {
        'task': 'lcc',
        'model': model_name,
        'em': em
    }
    dump_results(results, results_directory, results_filename)
    return results