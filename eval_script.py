import re
import json
from pathlib import Path
from datasets import load_dataset

from load_model_from_ckpt import load_model_from_ckpt
from eval_ppl import evaluate_ppl_red_pajamas


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def main():
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
    # ckpt_path = Path("/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B-copy/")
    ckpt_path = Path("/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_use_kv/")
    # ckpt_path = Path("/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_embed_only_use_kv_shuffle_old/")

    checkpoints = ckpt_path.glob("checkpoint*")
    checkpoints = sorted(
        ckpt_path.glob("checkpoint*"), key=lambda x: int(x.name.split("-")[-1])
    )

    eval_dataset = load_dataset("awettig/RedPajama-combined-15B-6K-llama", split="test")
    # eval_dataset = eval_dataset.shuffle(seed=42)
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

    save_jsonl(logs, "eval_results_LORA_kv.jsonl")
    # logs = [
    #     {"step": 5000, "val/av_loss": 1.7293137218864378},
    #     {"step": 10000, "val/av_loss": 1.677961277719723},
    #     {"step": 15000, "val/av_loss": 1.6701643097572374},
    #     {"step": 20000, "val/av_loss": 1.6709708926499083},
    # ]

    print(losses)
    print(1)
    pass


if __name__ == '__main__':
    main()