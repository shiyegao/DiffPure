#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=8 python generate.py --exp ./exp_results --config imagenet.yml \
  -i imagenetc \
  --t 150 \
  --adv_batch_size 1 \
  --domain imagenetc \
  --diffusion_type sde \
  --noise gaussian_noise
