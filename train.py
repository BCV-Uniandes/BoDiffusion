"""
Train a diffusion model on amass.
"""
import os
import json
import argparse
from collections import OrderedDict

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util_transformer import (
    create_model_condition_and_diffusion,
)
from guided_diffusion.train_util import TrainLoop

from utils import utils_option as option
from data.image_datasets import load_data_amass


def main():
    opt = create_opts()

    dist_util.setup_dist(devices=opt['gpu_ids'])
    logger.configure(dir=opt['path']['root'])

    with open(os.path.join(opt['path']['root'], 'options'), 'w') as f:
        json.dump(opt, f)

    logger.log("creating BoDiffusion...")
    model, diffusion = create_model_condition_and_diffusion(
        use_fp16=opt['fp16']['use_fp16'],
        **opt['ddpm'],
        **opt['diffusion'],
    )
    logger.log("** the model has " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + ' parameters **')
    logger.log("** the blocks have " + str(sum(p.numel() for p in model.blocks.parameters() if p.requires_grad)) + ' parameters **')
    model.to(dist_util.dev())
    
    if opt['fp16']['use_fp16']:
        model.convert_to_fp16()
    schedule_sampler = create_named_schedule_sampler(opt['train']['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data = load_data_amass(
        opt=opt,
        class_cond=opt['ddpm']['class_cond'],
        joint_cond=True,
        joint_cond_L=True,
    )
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=opt['datasets']['train']['dataloader_batch_size'],
        microbatch=opt['datasets']['train']['dataloader_microbatch'],
        lr=opt['train']['lr'],
        ema_rate=opt['train']['ema_rate'],
        log_interval=opt['train']['log_interval'],
        save_interval=opt['train']['save_interval'],
        resume_checkpoint=opt['path']['resume_checkpoint'],
        use_fp16=opt['fp16']['use_fp16'],
        fp16_scale_growth=opt['fp16']['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=opt['train']['weight_decay'],
        lr_anneal_steps=opt['train']['lr_anneal_steps'],
    ).run_loop()


def create_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train.json', help='Path to option JSON file.')

    json_str = ''
    with open(parser.parse_args().opt, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    return opt


if __name__ == "__main__":
    main()
