import os
import argparse
from collections import defaultdict
import wandb
import re

from setup_min_train import setup_training
from eval_ppl import evaluate_ppl_red_pajamas

from datasets import load_dataset


eval_batch_size = 5
max_samples = 30
pattern = re.compile(r"^chunk_\d+_(loss)$")
eval_ds = load_dataset("awettig/RedPajama-combined-15B-6K-llama", split="test")
split_size = 1024


step = 1


def main(args):
    project_name = "autocompressors"
    entity_name = "machine-learning-methods-in-software-engineering"

    config_path = "auco_min/config_min.yaml"
    (
        model,
        tokenizer,
        train_dataloader,
        optimizer,
        train_args,
        merge_config,
    ) = setup_training(config_path, args)
    run_name = (train_args.run_name).replace("LLaMA-1.3B", "MIN")
    batch_accum = train_args.total_batch_size
    step = 0
    av_loss = defaultdict(float)

    run = wandb.init(
        project=project_name, entity=entity_name, name=run_name, config=merge_config
    )

    for i, item in enumerate(train_dataloader):
        out = model(item["input_ids"])
        log = out.loss_log
        for key, value in log.items():
            av_loss[key] += value / batch_accum
        if i > 80:
            break
        if (i + 1) % batch_accum == 0:
            optimizer.step()
            # print(i + 1, av_loss["loss"])

            wandb.log(av_loss)
            av_loss = defaultdict(float)

            if step % train_args.save_steps == 0:
                output_dir = train_args.output_dir.replace("LLaMA-1.3B", "MIN")
                os.makedirs(output_dir, exist_ok=True)
                output_dir = os.path.join(output_dir, f"checkpoint-{step}")
                model.save_pretrained(output_dir)

                model.eval()
                eval_result = evaluate_ppl_red_pajamas(
                    model,
                    eval_ds,
                    eval_batch_size,
                    max_samples=max_samples,
                    split_size=split_size,
                    simple = True
                )

                wandb.log(eval_result)
                model.train()

            step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file")
    parser.add_argument("--suffix", help="Any suffix for run name")
    parser.add_argument(
        "--dev", action="store_true", help='Dev mode, adds "test" to the prefix'
    )
    args = parser.parse_args()
    main(args)
    pass
    print(1)
