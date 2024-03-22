### Install
Setup a new environment and install the most recent version of [pytorch](https://pytorch.org/),
followed by these libraries
```bash
  - pip install -U pip
  - pip install packaging
  - pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2
  - pip install -r requirements.txt
```

If you have CUDA <12, install
``` pip install flash-attn==2.3.1.post1 ```
https://github.com/Dao-AILab/flash-attention/issues/631#issuecomment-1820931085

### Training

`accelerate launch --config_file configs/accelerate_4gpu.yaml train.py --config configs/config.yaml` is the main method for training
AutoCompressors on multiple GPUs.
Parameters are set in the configs/config.yaml. Parameters for hardware are set in configs/accelerate_4gpu.yaml.
For training on single GPU and debugging you can run `python train.py`.
You can pass a suffix to the name of your run on wandb (as well as checkpoint folders) by --suffix your_suffix

### Eval on generation and loss

`CUDA_VISIBLE_DEVICES=1 python eval_lca_cc.py`

To run main you should pass checkpoint map file path (ckpt_map_path) that maps model name and checkpoint path.
If the model name starts with "base_model", then no checkpoint would be used, the basic (deepseek or LLaMA) model would be initialized
If the model name ends with `_some_integer`. Then `some_integer` would be used as the context size of the model. Otherwise, 6*1024 context would be used.
The length of the segment is hardcoded to be = 1024.
For the generation we fixed maximal length of the line = 128.

Another important parameters of the `eval_models_on_lcc` function is
`limit`: number of LCA samples used for generation.
`limit_loss_samples`: number of samples used for benchmarking evaluation
`results_path`: path to the file, where output results would be written (by appending new line ti existing)

`eval_models_on_lcc(ckpt_map_path, results_path, limit = 2000, limit_loss_samples = 1000)`
