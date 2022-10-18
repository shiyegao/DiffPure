# Diffusion Models for Adversarial Purification (Purification for ImageNet-C)

[Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2205.07460)


# Installation

## Checkpoint

Prepare [pretrained/guided_diffusion/256x256_diffusion_uncond.pt](https://github.com/openai/guided-diffusion#download-pre-trained-models).

## Dataset

Prepare [dataset/imagenetc_lmdb/val](https://github.com/hendrycks/robustness#imagenet-c).

## Package

Prepare pytorch and the following packages.

```bash

pip install git+https://github.com/RobustBench/robustbench.git lmdb torchsde Ninja
```

Ensure your ```libstdc++.so.6``` has 'GLIBCXX_3.4.29'.

```bash
rm /home/dqwang/anaconda3/envs/mim_gj/lib/libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/dqwang/anaconda3/envs/mim_gj/lib/libstdc++.so.6
```

# Usage

```bash
bash run_scripts/imagenet/generate.sh
```

## Citation

Please cite our paper, if you happen to use this codebase:

```
@inproceedings{nie2022DiffPure,
  title={Diffusion Models for Adversarial Purification},
  author={Nie, Weili and Guo, Brandon and Huang, Yujia and Xiao, Chaowei and Vahdat, Arash and Anandkumar, Anima},
  booktitle = {International Conference on Machine Learning (ICML)},
  year={2022}
}
```

