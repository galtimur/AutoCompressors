import torch
from transformers import LlamaTokenizer
from torch.optim import AdamW

from auco_min.data import get_dataloader
from auco_min.auco_min import ACCausalLM
from config_parser import parse_config

def setup_training(config_path, args):

    device = "cuda:0"
    model_args, data_args, training_args, merge_config = parse_config(config_path, args)
    model_args.use_kv = training_args.use_kv
    print(merge_config)

    split_size = 6 * 1024 // (training_args.training_substeps * training_args.segments_per_substep)
    # model
    model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = (
        ACCausalLM.from_pretrained(
            model_name,
            substeps=training_args.training_substeps,
            segments_per_substep=training_args.segments_per_substep,
            num_summary_vectors=training_args.summary_length,
            split_size=split_size,
            torch_dtype=torch.bfloat16
        )
        .to(device)
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters = {total_params:.0f} M")

    train_dataloader = get_dataloader(batch_size=training_args.per_device_train_batch_size, device=device)

    if model_args.lora:
        from peft import get_peft_model, LoraConfig, TaskType

        print(f"Building LoRA model")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            modules_to_save=model_args.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.train()
    if training_args.train_embed_only:
        num_trainable_parameters = 0
        for name, param in model.named_parameters():
            if 'embed_summary' in name.lower():
                param.requires_grad = True
                num_trainable_parameters += param.numel()
            else:
                param.requires_grad = False
        print(f"Number of trainable parameters = {num_trainable_parameters}")
        optimizer = torch.optim.AdamW(model.embed_summary.parameters(), lr=training_args.learning_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    return model, tokenizer, train_dataloader, optimizer, training_args, merge_config
