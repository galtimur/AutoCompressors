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