import json
from pathlib import Path

from fire import Fire
from huggingface_hub.utils import HFValidationError
from peft import PeftModel, get_peft_model, LoraConfig
import torch
from transformers import AutoConfig, AutoTokenizer

from auto_compressor import LlamaAutoCompressorModel


def check_existence_of_peft_checkpoint(directory: str | Path) -> bool:
    if (Path(directory) / 'adapter_config.json').exists():
        return True
    else:
        return False
    

def get_models_recursively(directory):
    models_dirs = []
    directory = Path(directory)
    for item in directory.glob('**'):
        if item.is_dir() and check_existence_of_peft_checkpoint(item):
            models_dirs.append(item)
    return models_dirs
            
    
def main(lora_path: str,
         save_dir: str,
         summary_length: int = 50,
         find_recursively: bool = False) -> None:
    lora_path = Path(lora_path).expanduser().resolve()
    save_dir = Path(save_dir).expanduser().resolve()
    
    if find_recursively:
        lora_paths = get_models_recursively(lora_path)
    else:
        lora_paths = [lora_path]
       
    print(f'Models to convert: {list(map(str, lora_paths))}')
    for lora_path in lora_paths:
        model_dir = save_dir / str(lora_path).split('/')[-1].lower().replace('-', '_')
        print(f'=== Converting {lora_path} to {model_dir} ===')
        with open(lora_path / 'adapter_config.json') as fp:
            base_model_name = json.load(fp)['base_model_name_or_path']
        
        config = AutoConfig.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(model_dir)
        
        config.summary_length = summary_length
        config.accumulate_summary = True
        config.segment_gradient_checkpointing = False
        
        model = LlamaAutoCompressorModel.from_pretrained(base_model_name, config=config, torch_dtype=torch.bfloat16)
        try:
            model = PeftModel.from_pretrained(model, lora_path)
        except HFValidationError as e:
            raise e
            print('Tried to load PEFT model but failed. Trying to load as a model+peft checkpoint')
            with open(lora_path / 'adapter_config.json') as fp:
                peft_config_dict = json.load(fp)
            print(json.dumps(peft_config_dict, indent=4))
            peft_config = LoraConfig(peft_config_dict)
            model = get_peft_model(model, peft_config)
            model.load_state_dict(torch.load(lora_path / 'pytorch_model.bin', map_location='cpu'))
        
        model = model.merge_and_unload()
        
        model.save_pretrained(model_dir, safe_serialization=True)
        
    
if __name__ == '__main__':
    Fire(main)
