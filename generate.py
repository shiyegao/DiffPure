# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import logging
import yaml
import os
import time
from tqdm import tqdm
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tvu
import utils
from utils import str2bool,  data_loader

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_sde import RevGuidedDiffusion



def generate(args, config):
    args.log_dir = os.path.join(args.image_folder, args.noise)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = utils.Logger(file_name=f'{args.log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    print('starting the model and loader...')
    diffusion = RevGuidedDiffusion(args, config, device=config.device)
    if ngpus > 1:
        diffusion = torch.nn.DataParallel(diffusion) 
    diffusion = diffusion.eval().to(config.device)
    
    # load data
    dataloader = data_loader(args, adv_batch_size)
    cnt = 0
    for i, (x, _, p) in tqdm(enumerate(dataloader)):

        # check image path
        p = p[0].split('/')
        save_dir = os.path.join(args.log_dir, p[-3], p[-2])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, p[-1])
        if os.path.exists(save_path):
            continue

        # generate image
        with torch.no_grad():
            x_re = diffusion.image_editing((x - 0.5) * 2)
            tvu.save_image((x_re[0] + 1) * 0.5, save_path)
            cnt += 1
    print(f"Generate {cnt}/{i+1}")
    
    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)

    parser.add_argument('--noise', type=str, default='gaussian_noise')
    # parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    generate(args, config)
