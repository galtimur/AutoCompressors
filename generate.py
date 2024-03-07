import json
from pathlib import Path
from transformers import LlamaForCausalLM
from load_model_from_ckpt import load_model_from_ckpt
import torch

device = "cuda:0"

def analyse_logits(model, output_ids, len_input):

    output_logits = model(output_ids, segment_lengths=1024).logits
    output_logits = output_logits[:, len_input:]
    output_ids = output_ids[:, len_input:]
    output_ids_test = torch.argmax(output_logits, dim=-1)
    verif_ids = output_ids[:, 1:] == output_ids_test[:, :-1]
    if torch.all(verif_ids):
        print("Generated most probable tokens")
    else:
        print(f"Generated {100 * torch.sum(verif_ids) / verif_ids.numel():.2f}% ({torch.sum(verif_ids)}/{verif_ids.numel()}) of probable tokens.")

        # indices of the tokens that did not matched
        false_indices = torch.nonzero(verif_ids == False).squeeze().to(device)
        if false_indices.ndim == 1:
            false_indices = false_indices.unsqueeze(0)
        # ids of the tokens that did not matched - from generation and
        false_logit_ids = output_ids_test[false_indices[:, 0], false_indices[:, 1]]
        true_logit_ids = output_ids[false_indices[:, 0], false_indices[:, 1]]
        # logits of the tokens that did not matched
        false_logits = output_logits[false_indices[:, 0], false_indices[:, 1], :]

        # values of the logits of the tokens that did not matched
        logits_false = false_logits[range(false_logit_ids.size(0)), false_logit_ids]
        logits_true = false_logits[range(true_logit_ids.size(0)), true_logit_ids]
        print("logits")
        print("original ids | model forward")
        print(torch.vstack([logits_true, logits_false]).t())

def main(checkpoint_path, prompt_filepath):

    model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
    model_base = LlamaForCausalLM.from_pretrained(model_name).to(device)
    model_base.eval()

    model, tokenizer, run_config = load_model_from_ckpt(checkpoint_path)
    model.use_kv = False  # We should rename past_kv if we want to use past_kv scheme
    model.eval()
    model.to(device)
    context_size = 6*1024
    segment_size = context_size // (
        run_config["training_substeps"] * run_config["segments_per_substep"]
    )

    model.gen_mode = True

    with open(prompt_filepath, "r") as f:
        prompt = f.read()
    # prompt = 400 * "In a hole in the ground there lived a"
    # prompt = 400 * "a=0 \n for i in range(10):\n    a+=1 \n"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids[:,:1838]
    len_input = input_ids.size(1)

    # output_ids_base = model_base.generate(input_ids, max_length=2000, do_sample=False, temperature=0)
    # generated_text_base = tokenizer.decode(output_ids_base[0][len_input:], skip_special_tokens=True)
    # print(generated_text_base)

    output_ids = model.generate(input_ids,
                                max_length=2000,
                                segment_lengths=1024,
                                do_sample=False)
    generated_text = tokenizer.decode(output_ids[0][len_input:], skip_special_tokens=True)
    analyse_logits(model, output_ids, len_input)

    print(generated_text)

    # model.embed_summary.weight = torch.nn.Parameter(-10000*model.embed_summary.weight)
    # model.embed_summary.reset_parameters()
    # output_ids = model.generate(input_ids,
    #                             max_length=2000,
    #                             segment_lengths=1024,
    #                             do_sample=False,
    #                             temperature=0)
    # generated_text_broken = tokenizer.decode(output_ids[0][len_input-5:], skip_special_tokens=True)
    # print(generated_text_broken)

if __name__ == "__main__":

    checkpoint_path = Path(
        # "/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_embed_only/checkpoint-46200"
        "/mnt/data2/galimzyanov/autocompressor/checkpoints/LLaMA-1.3B_sub3_seg2_sum50_new_split/checkpoint-17100"
    )  # 46200
    prompt_filepath = "out/passage_for_gen.txt"

    main(checkpoint_path, prompt_filepath)