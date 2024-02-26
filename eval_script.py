import re
import json
from pathlib import Path

from datasets import load_dataset
from fire import Fire

from load_model_from_ckpt import load_model_from_ckpt
from eval_ppl import evaluate_ppl_red_pajamas


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def main(shuffle: bool):
    device = "cuda:0"
    batch_size = 5
    max_samples = 300

    paths_to_check = {
        'Emb. only tune': Path('/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_embed_only'),
        'LoRA': Path('/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_old'),
        'KV cache LoRA': Path(''),
        'KV cache emb. only': Path(''),
    }

    pattern = r"checkpoint-(\d+)"
    for name, ckpt_path in paths_to_check.items():

        checkpoints = sorted(
            ckpt_path.glob("checkpoint*"), key=lambda x: int(x.name.split("-")[-1])
        )

        eval_dataset = load_dataset("awettig/RedPajama-combined-15B-6K-llama", split="test")
        if shuffle:
            eval_dataset = eval_dataset.shuffle(seed=42)
        example = next(iter(eval_dataset))
        context_size = len(example["labels"])

        losses = []
        logs = []
        # steps_to_do = [5000, 10000, 15000, 20000]
        steps_to_do = [5100, 9900, 15000, 20100]

        for checkpoint_path in checkpoints:
            match = re.search(pattern, checkpoint_path.name)
            step = int(match.group(1))
            if step not in steps_to_do:
                continue

            print(checkpoint_path.name)

            model, tokenizer, run_config = load_model_from_ckpt(checkpoint_path)
            model.eval()
            model.to(device)

            segment_size = context_size // (
                run_config["training_substeps"] * run_config["segments_per_substep"]
            )

            eval_result = evaluate_ppl_red_pajamas(
                model,
                eval_dataset,
                batch_size,
                max_samples=max_samples,
                split_size=segment_size,
                disable_tqdm=False,
            )
            eval_result["step"] = step
            av_loss = eval_result["total_loss"]
            print(av_loss)
            losses.append(av_loss)
            # logs.append({"step": step, "val/av_loss": av_loss})
            logs.append(eval_result)

            save_jsonl(logs, f"{name.replace('.', '').replace(' ', '_').lower()}.jsonl")


if __name__ == '__main__':
    Fire(main)