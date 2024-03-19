from utils_load_modules import get_run_mudules
import torch
import argparse
import time

def get_model(args, config_path):

    ModelTrainingModules = get_run_mudules(args, config_path)

    trainer, model_kwargs, train_dataset, val_dataset, merge_config = (
        ModelTrainingModules.trainer,
        ModelTrainingModules.model_kwargs,
        ModelTrainingModules.train_dataset,
        ModelTrainingModules.val_dataset,
        ModelTrainingModules.merge_config,
    )

    model = trainer.model

    return model, val_dataset

def generate_dummy_and_time(args, config_path, n_tokens = 128):

    model, val_dataset = get_model(args, config_path)

    example = torch.tensor(next(iter(val_dataset))["input_ids"]).unsqueeze(0).to(model.device)

    with torch.no_grad():
        start_time = time.time()
        out = model(example, segment_lengths=1024)
        init_time = start_time
        past_kv = out.past_key_values
        for i in range(n_tokens):
            out = model(example[:,:1], past_key_values=past_kv)
            past_kv = out.past_key_values
    pass
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the config file')
    parser.add_argument("--suffix", help="Any suffix for run name")
    parser.add_argument(
        "--dev", action="store_true", help='Dev mode, adds "test" to the prefix'
    )
    args = parser.parse_args()
    config_path = "configs/config_dev.yaml"

    main(args, config_path)
