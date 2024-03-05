import json
from pathlib import Path
from transformers import LlamaForCausalLM
from load_model_from_ckpt import load_model_from_ckpt
import torch

device = "cuda:0"
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
    # prompt = 300*"In a hole in the ground there lived a"

    with open(prompt_filepath, "r") as f:
        prompt = f.read()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    len_input = input_ids.size(1)

    output_ids_base = model_base.generate(input_ids, max_length=2000, do_sample=False, temperature=0)
    generated_text_base = tokenizer.decode(output_ids_base[0][len_input:], skip_special_tokens=True)
    print(generated_text_base)

    output_ids = model.generate(input_ids,
                                max_length=2000,
                                segment_lengths=1024,
                                do_sample=False,
                                temperature=0)
    generated_text = tokenizer.decode(output_ids[0][len_input:], skip_special_tokens=True)
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