#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=$1 python generate.py --exp /shared/dqwang/scratch/jingao/DiffPure --config imagenet.yml \
  -i imagenetcbar \
  --t 150 \
  --adv_batch_size 1 \
  --domain imagenetc \
  --diffusion_type sde \
  --noise $2

# 1    blue_noise_sample
# 2    brownish_noise
# 3    caustic_refraction
# 4    checkerboard_cutout
# 5    cocentric_sine_waves
# 6    inverse_sparkles
# 7    perlin_noise
# 8    plasma_noise
# 9    single_frequency_greyscale
# 10   sparkles