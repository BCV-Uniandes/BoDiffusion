"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
import json
import copy
import time
import logging
import argparse
import pickle
import numpy as np
import torch as th
import torch.distributed as dist
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util_transformer import (
    create_model_condition_and_diffusion,
)
from utils import utils_transform
from utils import utils_visualize as vis
from utils import utils_option as option
from utils import utils_logger

from itertools import product
from collections import OrderedDict
from pytorch_fid.fid_score import calculate_frechet_distance
from human_body_prior.body_model.body_model import BodyModel

from data.dataset_amass import AMASS_ALL_Dataset
from torch.utils.data import DataLoader
from data.select_dataset import define_Dataset
from models.select_model import define_Model

from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa
from guided_diffusion.respace import space_timesteps

from functorch import vmap

save_animation = False  # True
resolution = (500,500)

def sampling_all_video(
        diffusion,
        model,
        steps,
        joints,  # conditioning sequence [18 x T x 3]
        window_slide=20,
        ws=41,
        device='cpu',
        sampling_times=None,
    ):

    indices = list(range(steps))[::-1]
    time = joints.size(1)

    # generates noise of the shape of the sequence
    x = th.randn(6, time, 22, device=device)
    num_windows =  1 + (time - ws) // window_slide
    indexes = th.cat([th.arange(ws).view(1, -1)] * num_windows) + th.arange(num_windows).view(-1, 1) * window_slide
    indexes = indexes.to(device)

    slicer = vmap(lambda idx: joints[:, idx])
    cond = slicer(indexes).to(device)
    model_kwargs = {'joints': cond}
    model.precompute_joint_embedding(cond)

    for idx, t in tqdm(enumerate(indices)):

        # breaks x into sliding windows
        slicer = vmap(lambda idx: x[:, idx])
        data = slicer(indexes).squeeze(1)  # slicer works

        # filter the with the diffusion model
        t = th.tensor([t] * data.size(0), device=device)

        out = diffusion.p_mean_variance(
            model, data, t, model_kwargs=model_kwargs,
            clip_denoised=True
        )

        mean_mat = th.zeros(data.size(0), 6, time, 22, device=device)
        stdv_mat = th.zeros_like(mean_mat)
        ones_mat = th.zeros_like(mean_mat)

        for i in range(indexes.size(0)):
            mean_mat[i, :, indexes[i], :] = out['mean'][i]
            stdv_mat[i, :, indexes[i], :] = th.exp(out['log_variance'][i])
            ones_mat[i, :, indexes[i], :] = 1

        x = mean_mat.sum(dim=0) / ones_mat.sum(dim=0)
        std = stdv_mat.sum(dim=0).sqrt() / ones_mat.sum(dim=0)
        
        if (idx - 1) != len(indices) and idx > (len(indices) - 50):
            x += std * th.zeros_like(x)
        elif (idx - 1) != len(indices) and idx <= (len(indices) - 50):
            x += std * th.randn_like(x)

    return x

    
def process_and_save_output(data, root, name, save_vid, save_bm, time, body_model):
    '''
    :data: sample with shape FJ
    :root: saving root
    :generic name: name to save the data WITHOUT an extension
    :save_vid: create video from the predicted bm
    :save_bm: save the predicted body model
    '''
    data = data.unsqueeze(0)  # time now is 1
    data = data.permute((0, 2, 1))

    predicted_angle = utils_transform.sixd2aa(data, batch=True)
    predicted_angle = predicted_angle.reshape(time, -1)
    predicted_body = body_model(
        **{'pose_body':predicted_angle[...,3:66],
           'root_orient':predicted_angle[..., :3]}
    )
    position_global_full_local = predicted_body.Jtr[:,:22,:]
    t_head2root = position_global_full_local[:,15,:]
    t_root2world = -t_head2root
    # t_root2world = -t_head2root + t_head2world.cuda()

    predicted_data = {'pose_body':predicted_angle[...,3:66],
                      'root_orient':predicted_angle[...,:3],
                      'trans': t_root2world}
    predicted_body = body_model(**predicted_data)
    if save_vid:
        vis.save_animation(body_pose=predicted_body,
                           savepath=os.path.join(root, 'videos',
                                                 f'{name}.avi'),
                           bm=body_model, fps=60,
                           resolution=(800, 800))
    if save_bm:
        th.save(predicted_body,
                os.path.join(root, 'bmpth', f'{name}.pth'))

    predicted_data['position'] = predicted_body.Jtr[:, :22, :]
    return predicted_data


def process_and_save_output_all(data, root, name, save_vid, save_bm, time, body_model):
    '''
    :data: sample with shape FJ
    :root: saving root
    :generic name: name to save the data WITHOUT an extension
    :save_vid: create video from the predicted bm
    :save_bm: save the predicted body model
    '''
    data = data.permute((0, 3, 2, 1))

    predicted_angle = utils_transform.sixd2aa(data, batch=True)
    predicted_angle = predicted_angle.reshape(time, -1)
    predicted_body = body_model(
        **{'pose_body':predicted_angle[...,3:66],
           'root_orient':predicted_angle[..., :3]}
    )
    position_global_full_local = predicted_body.Jtr[:,:22,:]
    t_head2root = position_global_full_local[:,15,:]
    t_root2world = -t_head2root
    # t_root2world = -t_head2root + t_head2world.cuda()

    predicted_data = {'pose_body':predicted_angle[...,3:66],
                      'root_orient':predicted_angle[...,:3],
                      'trans': t_root2world}
    predicted_body = body_model(**predicted_data)
    if save_vid:
        vis.save_animation(body_pose=predicted_body,
                           savepath=os.path.join(root, 'videos',
                                                 f'{name}.avi'),
                           bm=body_model, fps=60,
                           resolution=(800, 800))
    if save_bm:
        th.save(predicted_body,
                os.path.join(root, 'bmpth', f'{name}.pth'))

    predicted_data['position'] = predicted_body.Jtr[:, :22, :]
    return predicted_data


def final_prediction(E, data, bm, gt=False, device=None):
    Head_trans_global = data['Head_trans_global'][:, :E.shape[0]].squeeze().to(device)
    # E = E['sample'].reshape(-1, 132)
    # E = data['H'].to(device)
    if not gt:
        E = E.permute((0, 2, 1))
    E = E.reshape(-1, 132)
    predicted_angle = utils_transform.sixd2aa(E[:,:132].reshape(-1,6).detach()).reshape(E[:,:132].shape[0],-1).float()

    # Calculate global translation

    T_head2world = Head_trans_global.clone()
    T_head2root_pred = th.eye(4).repeat(T_head2world.shape[0],1,1).cuda()
    rotation_local_matrot = aa2matrot(th.cat([th.zeros([predicted_angle.shape[0],3]).cuda(),predicted_angle[...,3:66]],dim=1).reshape(-1,3)).reshape(predicted_angle.shape[0],-1,9)
    rotation_global_matrot = local2global_pose(rotation_local_matrot, bm.kintree_table[0][:22].long())
    head2root_rotation = rotation_global_matrot[:,15,:]

    body_pose_local_pred=bm(**{'pose_body':predicted_angle[...,3:66]})
    head2root_translation = body_pose_local_pred.Jtr[:,15,:]
    T_head2root_pred[:,:3,:3] = head2root_rotation
    T_head2root_pred[:,:3,3] = head2root_translation
    t_head2world = T_head2world[:,:3,3].clone()
    T_head2world[:,:3,3] = 0
    T_root2world_pred = th.matmul(T_head2world, th.inverse(T_head2root_pred))

    rotation_root2world_pred = matrot2aa(T_root2world_pred[:,:3,:3])
    translation_root2world_pred = T_root2world_pred[:,:3,3]
    body_pose_local=bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3]})
    position_global_full_local = body_pose_local.Jtr[:,:22,:]
    t_head2root = position_global_full_local[:,15,:]
    t_root2world = -t_head2root+t_head2world.cuda()

    predicted_body=bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': t_root2world}) 
    # No stabilizer: 'root_orient':rotation_root2world_pred.cuda()

    predicted_position = predicted_body.Jtr[:,:22,:]
    
    predicted_translation = t_root2world

    body_parms = OrderedDict()
    body_parms['pose_body'] = predicted_angle[...,3:66]
    body_parms['root_orient'] = predicted_angle[...,:3]
    body_parms['trans'] = predicted_translation
    body_parms['position'] = predicted_position
    body_parms['body'] = predicted_body

    return body_parms


class DataSlicer():
    def __init__(self, window_size, data):
        
        self.ws = window_size
        self.data = data
        
    def __len__(self):
        return len(self.data) - self.ws - 1

    def __getitem__(self, idx):
        data = self.data[idx:(idx + self.ws)]
        inpt = data[-self.ws:].permute((1, 0, 2)).squeeze(1)  # when it is t=1, we remove first dim
        return inpt 
        
def compute_mean_and_cov(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def main():
    opt = create_opts()

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter
    
    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger_transformer = logging.getLogger(logger_name)

    dist_util.setup_dist(devices=opt['gpu_ids'])
    logger.configure(dir=opt['path']['root'])

    logger.log("creating BoDiffusion...")
    
    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt['datasets']['test']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,  # dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=8,
                                     drop_last=False, pin_memory=True)
        elif phase == 'train':
            continue
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # Model Initialization
    model, diffusion = create_model_condition_and_diffusion(
        use_fp16=opt['fp16']['use_fp16'],
        **opt['ddpm'],
        **opt['diffusion']
    )
    chpnt = opt['path']['resume_checkpoint']
    print(f'resuming checkpoint from {chpnt}')
    model.load_state_dict(
        dist_util.load_state_dict(
            os.path.join(opt['path']['resume_checkpoint']),
            map_location="cpu"
        )
    )
    model.to(dist_util.dev())
    if opt['fp16']['use_fp16']:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    shape = (
        opt['datasets']['test']['dataloader_batch_size'],
        opt['ddpm']['in_channels'],
        *opt['ddpm']['image_size']
    )
    # import pdb; pdb.set_trace()
    os.makedirs(os.path.join(opt['path']['root'], 'videos'), exist_ok=True)
    os.makedirs(os.path.join(opt['path']['root'], 'denoise', 'videos'), exist_ok=True)

    # instantiate the BodyModel
    subject_gender = "male"
    bm_fname = os.path.join(opt['support_dir'], 'body_models/smplh/{}/model.npz'.format(subject_gender))
    dmpl_fname = os.path.join(opt['support_dir'], 'body_models/dmpls/{}/model.npz'.format(subject_gender))
    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    body_model = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(dist_util.dev())

    bs = opt['datasets']['test']['dataloader_batch_size']
    idx = 0

    error_stats = {
        'rot_error': [],
        'pos_error': [],
        'vel_error': [],
        'pos_error_hands': [],
        'rot_error_hands_and_head': [],
        'pos_error_upper': [],
        'pos_error_lower': [],
        'pos_error_pelvis': [],
    }
    logger.log('Evaluating {} times per timestep'.format(opt['num_evaluation']))

    dist.barrier()

    # ============================================================================================================
    # COMPUTE FID AND OTHER METRICS FROM INSTANCES
    # ============================================================================================================

    window_slide = 20
    test_times = [999]
    if opt['diffusion']['timestep_respacing'] != '':
        if opt['diffusion']['timestep_respacing'].startswith("ddim"):
            timestep_respacing = int(opt['diffusion']['timestep_respacing'][len("ddim") :])
        else:
            timestep_respacing = opt['diffusion']['timestep_respacing']
        test_times = [int(i * int(timestep_respacing) / int(opt['diffusion']['diffusion_steps'])) for i in test_times]

    # load amass dataset
    dataset = AMASS_ALL_Dataset(opt=opt['datasets']['test'])
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=10,
        drop_last=True
    )
    
    results = {i: [copy.deepcopy(error_stats) for _ in range(opt['num_evaluation'])] for i in test_times}
    pr_arr = {i: [] for i in test_times}
    rd_arr = {i: [] for i in test_times}
    max_ws = max(opt['datasets']['test']['window_size'], opt['datasets']['test']['cond_window_size'])
    ws = opt['datasets']['test']['window_size']
    cws = opt['datasets']['test']['cond_window_size']

    if not opt['diffusion']['timestep_respacing']:
        respacing_mode = test_times
    else: 
        respacing_mode = opt['diffusion']['timestep_respacing']
    sampling_times = space_timesteps(opt['diffusion']['diffusion_steps'], respacing_mode)
    sampling_times = sorted(sampling_times)

    for i in range(opt['num_evaluation']):
        start = time.time()
        for steps in test_times:
            vidx = 0
            rot_error = []
            pos_error = []
            error_dir = os.path.join(opt['path']['task'],'errors',opt['identifier'])
            # if not os.path.exists(error_dir):
            #     os.makedirs(error_dir)   
            preds_dir = os.path.join(opt['path']['task'],'preds',opt['identifier'])
            # if not os.path.exists(preds_dir):
            #     os.makedirs(preds_dir)             
            for idx, test_data in enumerate(test_loader):
                filename_ = os.path.join(error_dir,str(idx)+'.pkl')
                filename_preds = os.path.join(preds_dir,str(idx)+'.pkl')

                if os.path.isfile(filename_):
                    print(f'ID {idx} already has a .pkl file')
                    continue 

                if test_data['H'].shape[1] < max_ws:
                    continue

                cond = test_data['L'].to(dist_util.dev())
                cond = th.permute(cond.squeeze(0).reshape(-1, 3, 18), (2, 0, 1))

                Time = cond.size(1)
                new_time = Time - ((Time - ws) % window_slide)

                ground_truth = test_data['H'].to(dist_util.dev())  # .squeeze(0)
                
                cond = cond[:, :new_time]
                ground_truth = ground_truth[:, :new_time]

                pred = sampling_all_video(
                    diffusion,
                    model,
                    steps,
                    cond,  # conditioning sequence [18 x T x 3]
                    window_slide=window_slide,
                    ws=ws,
                    device=dist_util.dev(),
                    sampling_times=sampling_times,
                )
                
                # =======================================
                # process all frames
                # =======================================

                pred = th.permute(pred, (1, 0, 2))
                ground_truth = ground_truth.squeeze(0)
                t = th.tensor([steps] * pred.size(0), device=pred.device)
                noised = diffusion.q_sample(pred.detach(), t, noise=th.randn_like(pred))
                Time = pred.size(0)

                pr = final_prediction(pred, test_data,body_model, device=pred.device)
                gt = final_prediction(ground_truth, test_data,body_model, True, pred.device)

                gt_save = gt.copy()
                gt_save.pop('body')
                pr_save = pr.copy()
                pr_save.pop('body')
                gt_save['filename'] = test_data['filename']

                with open(filename_preds, 'wb') as f:
                    pickle.dump([pr_save,gt_save], f)

                vidx += 1
                del gt_save
                del pr_save

                gt_angle = gt['pose_body'].reshape(-1, 21, 3)
                pr_angle = pr['pose_body'].reshape(-1, 21, 3)

                gt_pos = gt['position'].reshape(-1, 22, 3)
                pr_pos = pr['position'].reshape(-1, 22, 3)

                gt_vel = (gt_pos[1:, ...] - gt_pos[:-1, ...]) * 60
                pr_vel = (pr_pos[1:, ...] - pr_pos[:-1, ...]) * 60

                rot_error_ = th.mean(th.absolute(gt_angle - pr_angle)).cpu()
                rot_error_hands_and_head_ = th.mean(th.absolute(gt_angle - pr_angle)[:, [15 - 1, 20 - 1, 21 - 1], :]).cpu()
                pos_error_ = th.mean(th.sqrt(th.sum(th.square(gt_pos - pr_pos), axis=-1))).cpu()
                pos_error_hands_ = th.mean(th.sqrt(th.sum(th.square(gt_pos - pr_pos), axis=-1))[...,[20, 21]]).cpu()
                vel_error_ = th.mean(th.sqrt(th.sum(th.square(gt_vel - pr_vel), axis=-1))).cpu()
                pos_error_upper_ = th.mean(th.sqrt(th.sum(th.square(gt_pos-pr_pos), axis=-1))[...,[3,6,9,12,13,14,15,16,17,18,19,20,21]]).cpu()
                pos_error_lower_ = th.mean(th.sqrt(th.sum(th.square(gt_pos-pr_pos), axis=-1))[...,[1,2,4,5,7,8,10,11]]).cpu()
                pos_error_pelvis_ = th.mean(th.sqrt(th.sum(th.square(gt_pos-pr_pos), axis=-1))[...,[0]]).cpu()

                indv_errors = {
                    'rot_error': rot_error_,
                    'pos_error': pos_error_,
                    'vel_error': vel_error_,
                    'pos_error_hands': pos_error_hands_,
                    'noise_rot_error': rot_error_nn_,
                    'rot_error_hands_and_head': rot_error_hands_and_head_,
                    'pos_error_upper': pos_error_upper_,
                    'pos_error_lower': pos_error_lower_,
                    'pos_error_pelvis': pos_error_pelvis_,

                }     
                with open(filename_, 'wb') as f:
                    pickle.dump(indv_errors, f)

                results[steps][i]['rot_error'].append(rot_error_)
                results[steps][i]['pos_error'].append(pos_error_)
                results[steps][i]['vel_error'].append(vel_error_)
                results[steps][i]['pos_error_hands'].append(pos_error_hands_)
                results[steps][i]['rot_error_hands_and_head'].append(rot_error_hands_and_head_)
                pr_arr[steps].append(pr['pose_body'].cpu().detach())
                rd_arr[steps].append(nn['pose_body'].cpu().detach())

                results[steps][i]['pos_error_upper'].append(pos_error_upper_)
                results[steps][i]['pos_error_lower'].append(pos_error_lower_)
                results[steps][i]['pos_error_pelvis'].append(pos_error_pelvis_)

                if idx in [0, 3, 4, 5, 6, 10, 20, 50] and save_animation:
                    video_dir = os.path.join(opt['path']['images'],opt['identifier'], str(idx))
                    if not os.path.exists(video_dir):
                        os.makedirs(video_dir)
                    
                    save_video_path = os.path.join(video_dir, '{:d}.avi'.format(current_step))
                    vis.save_animation_gt(gt['body'], predicted_body, nn['body'], pr['body'], savepath=save_video_path, bm = body_model, text=opt['text'], fps=60, resolution = resolution)
                print('Iteration [{}] | Sample [{} / {}] | Steps [{}] | Time [{:<.5f}]'.format(i + 1, idx + 1, len(loader), steps, time.time() - start))
                if idx % 10 == 0:
                    # logger_transformer.info(f'rot_error: {sum(rot_error) / len(rot_error) * 57.2958}, pos_error: {sum(pos_error) / len(pos_error)*100}')
                    logger_transformer.info(
                    f"Average errors for iter {idx} steps {steps}"
                    )
                    logger_transformer.info(
                    "Average rotational error [degree]: {:<.5f}, H+Hs rotational error [degree]: {:<.5f}, Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}, Average positional error at hand [cm]: {:<.5f}, Avpos error at upper [cm]: {:<.5f}, Avpos error at lower [cm]: {:<.5f}, Avpos error at pelvis [cm]: {:<.5f}\n".format(
                        sum(results[steps][i]['rot_error']) / len(results[steps][i]['rot_error']) * 57.2958, sum(results[steps][i]['rot_error_hands_and_head']) / len(results[steps][i]['rot_error_hands_and_head']) * 57.2958, sum(results[steps][i]['pos_error']) / len(results[steps][i]['pos_error']) * 100, sum(results[steps][i]['vel_error']) / len(results[steps][i]['vel_error']) * 100, sum(results[steps][i]['pos_error_hands']) / len(results[steps][i]['pos_error_hands']) * 100, sum(results[steps][i]['pos_error_upper']) / len(results[steps][i]['pos_error_upper']) * 100, sum(results[steps][i]['pos_error_lower']) / len(results[steps][i]['pos_error_lower']) * 100, sum(results[steps][i]['pos_error_pelvis']) / len(results[steps][i]['pos_error_pelvis']) * 100
                    )
                    )

        logger.log('Done evaluation [{}] | Steps [{}] | Time [{:<.5f}]'.format(i + 1, steps, time.time() - start))
        logger.log('=' * 75)

    logger.log('Finished evaluation')

    for steps, i in product(test_times, range(opt['num_evaluation'])):
        results[steps][i]['rot_error'] = sum(results[steps][i]['rot_error'])/len(results[steps][i]['rot_error'])
        results[steps][i]['pos_error'] = sum(results[steps][i]['pos_error'])/len(results[steps][i]['pos_error'])
        results[steps][i]['vel_error'] = sum(results[steps][i]['vel_error'])/len(results[steps][i]['vel_error'])
        results[steps][i]['pos_error_hands'] = sum(results[steps][i]['pos_error_hands'])/len(results[steps][i]['pos_error_hands'])
        results[steps][i]['rot_error_hands_and_head'] = sum(results[steps][i]['rot_error_hands_and_head'])/len(results[steps][i]['rot_error_hands_and_head'])
        results[steps][i]['pos_error_upper'] = sum(results[steps][i]['pos_error_upper'])/len(results[steps][i]['pos_error_upper'])
        results[steps][i]['pos_error_lower'] = sum(results[steps][i]['pos_error_lower'])/len(results[steps][i]['pos_error_lower'])
        results[steps][i]['pos_error_pelvis'] = sum(results[steps][i]['pos_error_pelvis'])/len(results[steps][i]['pos_error_pelvis'])

    # compute final results
    final_results = {t: copy.deepcopy(error_stats) for t in test_times}
    for steps, k in product(test_times, ['rot_error', 'pos_error', 'vel_error', 'pos_error_hands', 'noise_rot_error', 'rot_error_hands_and_head', 'noise_pos_error','noise_vel_error','pos_error_upper','pos_error_lower','pos_error_pelvis','pos_error_nn_hands','pos_error_nn_upper','pos_error_nn_lower','pos_error_nn_pelvis']):
        factor = 57.2958 if k in ['rot_error', 'noise_rot_error', 'rot_error_hands_and_head'] else 100
        final_results[steps][k] = factor * sum([results[steps][i][k] for i in range(opt['num_evaluation'])]) / opt['num_evaluation']

    for steps in test_times:
        logger.log('+' * 50)
        logger.log(f'For #steps = {steps}')
        logger.log('Average rotational error [degree]: {:<.5f}'.format(final_results[steps]['rot_error']))
        logger.log('Average rotational error (hands and head) [degree]: {:<.5f}'.format(final_results[steps]['rot_error_hands_and_head']))
        logger.log('Average rotational error (noise) [degree]: {:<.5f}'.format(final_results[steps]['noise_rot_error']))
        logger.log('Average positional error [cm]: {:<.5f}'.format(final_results[steps]['pos_error']))
        logger.log('Average velocity error [cm/s]: {:<.5f}'.format(final_results[steps]['vel_error']))
        logger.log('Average positional error at hands [cm]: {:<.5f}'.format(final_results[steps]['pos_error_hands']))
        logger.log('Average positional error at upper [cm]: {:<.5f}'.format(final_results[steps]['pos_error_upper']))
        logger.log('Average positional error at lower [cm]: {:<.5f}'.format(final_results[steps]['pos_error_lower']))
        logger.log('Average positional error at pelvis [cm]: {:<.5f}'.format(final_results[steps]['pos_error_pelvis']))


def create_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/test.json',
                        help='Path to option JSON file.')
    parser.add_argument('-timestep_respacing', type=str, default='',
                        help='time respacing to speed-up sampling')
    parser.add_argument('-resume_checkpoint', type=str, default='results/model600000.pt',
                        help='Model pt weights')
    parser.add_argument('-use_ddim', action='store_true',
                        help='Sampling using the ddpm algorithm')
    parser.add_argument('-clip_denoised', action='store_true',
                        help='Clip the noise at sampling')
    parser.add_argument('-num_evaluation', type=int, default=1,
                        help='Number of test dataset loops')
    parser.add_argument('-save_vids', type=int, default=0,
                        help='Store videos')
    parser.add_argument('-time_inpainting', action='store_true', default=None,
                        help='Inpaining frames')
    parser.add_argument('-glide', action='store_true', help='Choose either RP of GL for GT noise')                    
    parser.add_argument('-guidance', nargs='+', type=int, default=[],
                            help='Inpaining joints')
    parser.add_argument('-gpu_id', type=str, default="0", help='gpu id')
    parser.add_argument('-steps', type=int, default=1, help='num steps for de DDPM')
    parser.add_argument('-identifier', type=str, default='', help='identifier for the experiment')
    parser.add_argument('-min_id', type=int, default=0, help='min_id for running in -paralel- (inclusive)')
    parser.add_argument('-max_id', type=int, default=536, help='max_id for running in -paralel- (inclusive)')

    args = parser.parse_args()
    json_str = ''
    with open(args.opt, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    opt = option.parse(args.opt, args.gpu_id, is_train=True)
    opt['diffusion']['timestep_respacing'] = args.timestep_respacing
    opt['path']['resume_checkpoint'] = args.resume_checkpoint
    opt['use_ddim'] = args.use_ddim
    opt['clip_denoised'] = args.clip_denoised
    opt['num_evaluation'] = args.num_evaluation
    opt['save_vid'] = args.save_vids
    opt['steps'] = args.steps
    opt['identifier'] = args.identifier

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    return opt

if __name__ == "__main__":
    with th.inference_mode():  # allow even faster computations
        main()