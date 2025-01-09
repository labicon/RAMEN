import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import json


from torch.utils.data import DataLoader
from tqdm import tqdm, trange


# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from model.decoder_NICESLAM import NICE
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion

import sys

from torch.nn.utils import parameters_to_vector as p2v
import copy

import cv2

import redis # for reading images from ROS2 nodes
import pickle # for loading data from redis
from scipy.spatial.transform import Rotation # to process quaternion from vicon bridge 


class Mapping():
    def __init__(self, config, id, dataset_info):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = id 
        self.dataset_info = dataset_info 

        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)
        self.fix_decoder = config['multi_agents']['fix_decoder']
        self.create_optimizer()
      
        self.dist_algorithm = config['multi_agents']['distributed_algorithm']
        self.track_uncertainty = config['multi_agents']['track_uncertainty']
        if self.track_uncertainty:
            self.uncertainty_tensor = torch.zeros(self.model.embed_fn.params.size()).to(self.device)
            self.W_i = torch.zeros(self.uncertainty_tensor.size()).to(self.device) 

        # save loss 
        self.total_loss = []
        self.obj_loss = []
        self.lag_loss = []
        self.aug_loss = []

        # initialize dual variable 
        theta_i = p2v(self.model.parameters())
        self.p_i = torch.zeros(theta_i.size()).to(self.device)
        # a list to hold neighbor model parameters, and uncertainty tensor (optional)
        self.neighbors = []
        # step size in the gradient ascent of the dual variable
        self.rho = config['multi_agents']['rho']

        # for DSGD/DSGT
        self.ds_mat = None # doubly stochastic matrix for DSGD/DSGT
        self.num_params = sum( p.numel() for p in self.model.parameters() )
        self.alpha = config['multi_agents']['alpha']
        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.model.parameters()
        ]
        self.g_dsgt = copy.deepcopy(base_zeros)
        self.y_dsgt = copy.deepcopy(base_zeros) 




    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        

    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
    

    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(torch.float32).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(torch.float32).to(self.device)


    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset_info['num_frames'] // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset_info['num_rays_to_save'])
        return KeyFrameDatabase(config, 
                                self.dataset_info['H'], 
                                self.dataset_info['W'], 
                                num_kf, 
                                self.dataset_info['num_rays_to_save'], 
                                self.device)
 

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))


    def load_decoder(self, load_path):
        dict = torch.load(load_path, weights_only=True)
        model_dict = dict['model']
        del model_dict['embedpos_fn.params']
        del model_dict['embed_fn.params']
        self.model.load_state_dict(model_dict, strict=False) # load from a partial state_dict missing some keys, use strict=False
    

    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'total_loss': self.total_loss,
                     'obj_loss': self.obj_loss,
                     'lag_loss': self.lag_loss,
                     'aug_loss': self.aug_loss,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')


    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']


    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice


    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        
        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'], 
                                                                                  margin=self.config['training']['smooth_margin'])
        
        return loss             


    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        print(f'Agent {self.agent_id} First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset_info['H'], self.dataset_info['W'], self.config['mapping']['sample'])
            
            indice_h, indice_w = indice % (self.dataset_info['H']), indice // (self.dataset_info['H'])
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()

            if self.track_uncertainty:
                grid_grad = self.model.embed_fn.params.grad
                grid_has_grad = (torch.abs(grid_grad) > 0).to(torch.int32)
                self.uncertainty_tensor += grid_has_grad

            self.map_optimizer.step()

        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)
        
        print(f'Agent {self.agent_id} First frame mapping done')
        return ret, loss


    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
    
    
    def scaling_AUQ_CADMM(self, k, uncertainty_i, uncertainty_j):
        uncertainty = uncertainty_i + uncertainty_j
        a_1 = self.rho/1000
        b_1 = self.rho

        # scale to a_1 and b_1: uncertainty_scaled = p*uncertainty + q
        p = (b_1-a_1)/(torch.max(uncertainty) - torch.min(uncertainty)) 
        q = a_1 - p*torch.min(uncertainty)
        return p, q


    def communicate(self,input):
        model_j = input[0]
        theta_j = p2v(model_j.parameters()).detach()

        if self.dist_algorithm == 'AUQ_CADMM':
            uncertainty_j = input[1].detach()
            step = input[2]
            self.neighbors.append( [theta_j, uncertainty_j] )

        elif self.dist_algorithm in ('CADMM', 'MACIM'):
            self.neighbors.append( [theta_j] )

        elif self.dist_algorithm == 'DSGD':
            j = input[1]
            self.neighbors.append( [model_j.parameters(), j] )

        elif self.dist_algorithm == 'DSGT':
            y_dsgt_j = input[1]
            j = input[2]
            self.neighbors.append( [model_j.parameters(), y_dsgt_j, j] )


    def dual_update(self, theta_i_k):
        for neighbor in self.neighbors:
            theta_j_k = neighbor[0]
            self.p_i += self.rho * (theta_i_k - theta_j_k)    


    def dual_update_AUQ_CADMM(self, theta_i_k, uncertainty_i, k):
        padding_size = theta_i_k.size(0) - uncertainty_i.size(0)
        for neighbor in self.neighbors:
            theta_j_k = neighbor[0]
            uncertainty_j = neighbor[1]
            p, q = self.scaling_AUQ_CADMM(k, uncertainty_i, uncertainty_j)
            W_i = p*uncertainty_i + q
            W_i = torch.nn.functional.pad(W_i, (0,padding_size), "constant", self.rho) 
            W_j = p*uncertainty_j + q
            W_j = torch.nn.functional.pad(W_j, (0,padding_size), "constant", self.rho) 
            self.p_i +=  2*W_i * torch.div( W_j*theta_i_k - W_j*theta_j_k, W_i + W_j)


    def primal_update(self, theta_i_k, loss):
        theta_i = p2v(self.model.parameters())
        lag_loss = torch.dot(theta_i, self.p_i) #TODO: uncomment? comment?
        aug_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
        for neighbor in self.neighbors:
            theta_j_k = neighbor[0]
            aug_loss += self.rho * torch.norm(theta_i - (theta_i_k+theta_j_k)/2)**2
        loss += lag_loss + aug_loss 
        return loss, lag_loss.item(), aug_loss.item()
    

    def primal_update_AUQ_CADMM(self, theta_i_k, loss, uncertainty_i, k):
        theta_i = p2v(self.model.parameters())
        lag_loss = torch.dot(theta_i, self.p_i) #TODO: uncomment? comment?
        aug_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
        padding_size = theta_i.size(0) - uncertainty_i.size(0)
        for neighbor in self.neighbors:
            theta_j_k = neighbor[0]     
            uncertainty_j = neighbor[1]
            p, q = self.scaling_AUQ_CADMM(k, uncertainty_i, uncertainty_j)
            W_i = p*uncertainty_i + q
            W_i = torch.nn.functional.pad(W_i, (0,padding_size), "constant", self.rho) 
            W_j = p*uncertainty_j + q
            W_j = torch.nn.functional.pad(W_j, (0,padding_size), "constant", self.rho) 
            difference = theta_i - torch.div( W_i*theta_i_k + W_j*theta_j_k, W_i + W_j)
            weighted_norm = torch.dot(difference*W_i, difference)
            aug_loss += weighted_norm
        loss += lag_loss + aug_loss
        return loss, lag_loss.item(), aug_loss.item()


    def MACIM_cc_loss(self, loss):
        theta_i = p2v(self.model.parameters())
        for neighbor in self.neighbors:
            theta_j = neighbor[0]
            difference = self.rho * torch.norm(theta_i - theta_j)**2
            loss += difference
        return loss


    def DSGD_update(self):
        rid = self.agent_id
        deg_i = len(self.neighbors)
        w = 1/(deg_i+1)
        with torch.no_grad():
            for param_i in self.model.parameters():
                #param_i.multiply_(self.ds_mat[rid, rid]) 
                param_i.multiply_(w) 
                param_i.add_(-self.alpha * param_i.grad)  # Gradient descent update
                param_i.grad.zero_()  # Reset the gradient

            for model_j, j in self.neighbors:
                for param_i, param_j in zip(self.model.parameters(), model_j):
                    #param_i.add_(self.ds_mat[rid, j] * param_j)
                    param_i.add_(w * param_j)


    def DSGT_update(self):
        rid = self.agent_id
        deg_i = len(self.neighbors)
        w = 1/(deg_i+1)
        with torch.no_grad():
            for p, param_i in enumerate(self.model.parameters()):
                param_i.multiply_(w) 
                param_i.add_(-w*self.alpha*self.y_dsgt[p])  # Gradient descent update
                self.y_dsgt[p].multiply_(w) 
                self.y_dsgt[p].add_(param_i.grad - self.g_dsgt[p]) 
                self.g_dsgt[p] = param_i.grad.clone()
                param_i.grad.zero_() 

            for model_j, y_j, j in self.neighbors:
                for p, (param_i, param_j) in enumerate(zip(self.model.parameters(), model_j)):
                    param_i.add_(w*param_j - w*self.alpha*y_j[p])
                    self.y_dsgt[p].add_(w*y_j[p])
               

    def global_BA(self, batch, cur_frame_id, dist_algorithm):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
            dist_algorithm: algorithm used for multi-agent learning
        '''

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
        current_pose = self.est_c2w_data[cur_frame_id][None,...]
        poses_all = torch.cat([poses_fixed, current_pose], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1]) 

        theta_i_k = p2v(self.model.parameters()).detach()


        if dist_algorithm == 'CADMM':
            self.dual_update(theta_i_k) 
        elif dist_algorithm == 'AUQ_CADMM':
            self.dual_update_AUQ_CADMM(theta_i_k, self.uncertainty_tensor, cur_frame_id)

        mean_total_loss = 0
        mean_obj_loss = 0
        mean_lag_loss = 0
        mean_aug_loss = 0
        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset_info['H'] * self.dataset_info['W']),max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            self.map_optimizer.zero_grad()

            loss = self.get_loss_from_ret(ret, smooth=True)
            loss.backward(retain_graph=True)
            mean_obj_loss += loss.item() #item() method extracts the loss’s value as a Python float.

            if self.track_uncertainty:
                grid_grad = self.model.embed_fn.params.grad
                grid_has_grad = (torch.abs(grid_grad) > 0).to(torch.int32)
                self.uncertainty_tensor += grid_has_grad

            if dist_algorithm == 'CADMM':
                loss, lag_loss, aug_loss = self.primal_update(theta_i_k, loss)
                loss.backward(retain_graph=True)
                self.map_optimizer.step()
                mean_lag_loss += lag_loss
                mean_aug_loss += aug_loss

            elif dist_algorithm == 'AUQ_CADMM':
                loss, lag_loss, aug_loss  = self.primal_update_AUQ_CADMM(theta_i_k, loss, self.uncertainty_tensor, cur_frame_id)
                loss.backward(retain_graph=True)
                self.map_optimizer.step()
                mean_lag_loss += lag_loss
                mean_aug_loss += aug_loss

            elif dist_algorithm == 'MACIM':
                loss = self.MACIM_cc_loss(loss)
                loss.backward(retain_graph=True)
                self.map_optimizer.step()

            elif dist_algorithm == 'DSGD':
                self.DSGD_update()
                break # DSDG does one update per mapping iteration 

            elif dist_algorithm == 'DSGT':
                self.DSGT_update()
                break # DSDT does one update per mapping iteration 


            mean_total_loss += loss.item()


        # save loss info 
        mean_total_loss /= self.config['mapping']['iters']
        mean_obj_loss /= self.config['mapping']['iters']
        mean_lag_loss /= self.config['mapping']['iters']
        mean_aug_loss /= self.config['mapping']['iters']
        self.total_loss.append( mean_total_loss )
        self.obj_loss.append(mean_obj_loss)
        self.lag_loss.append(mean_lag_loss)
        self.aug_loss.append(mean_aug_loss)


    def tracking_render(self, batch, frame_id):
        '''
            just save ground truth pose
        '''
        c2w_gt = batch['c2w'][0].to(self.device)
        self.est_c2w_data[frame_id] = c2w_gt


    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        if self.fix_decoder:
            #TODO: pretrain
            trainable_parameters = [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
        else:
            # Optimizer for BA
            trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                    {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]

        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        
    
    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=color_func, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)    

        if self.track_uncertainty == True:
            uncertainty_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'uncertain_track{}.pt'.format(i))
            torch.save(self.uncertainty_tensor, uncertainty_savepath)


    def run(self, i, batch):
        """
            @param i: current step
            @param batch:
        """
        # First frame mapping
        if i == 0:
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            return 
        
        # Tracking + Mapping
        self.tracking_render(batch, i)

        if i%self.config['mapping']['map_every']==0:
            self.global_BA(batch, i, self.dist_algorithm)
            
        # Add keyframe
        if i % self.config['mapping']['keyframe_every'] == 0:
            self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
            #print(f'\nAgent {self.agent_id} add keyframe:{i}')

        if i % self.config['mesh']['vis']==0:
            model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'checkpoint_{}.pt'.format(i)) 
            self.save_ckpt(model_savepath)
            self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])








def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type ==  'OpenGL':
        dirs = torch.stack([(i - cx)/fx, -(j - cy)/fy, -torch.ones_like(i)], -1)
    elif type == 'OpenCV':
        dirs = torch.stack([(i - cx)/fx, (j - cy)/fy, torch.ones_like(i)], -1)
    else:
        raise NotImplementedError()

    rays_d = dirs
    return rays_d




def data_loading(redis_client, max_depth, rays_d, step):
    # Get RGB image from Redis
    rgb_data = redis_client.get('rgb_image')
    if rgb_data:
        rgb_image = pickle.loads(rgb_data)
        rgb_image = torch.from_numpy(rgb_image.astype(np.float32) / 255.) # Normalize to [0, 1]

    # Get depth image from Redis
    depth_data = redis_client.get('depth_image')
    if depth_data:
        depth_image = pickle.loads(depth_data)
        depth_image = torch.from_numpy(depth_image.astype(np.float32) / 1000.) # mm to m

    # Get pose from Redis 
    pickled_vicon_data = redis_client.get('vicon_data') 
    if pickled_vicon_data:
        vicon_data = pickle.loads(pickled_vicon_data)  # vicon_data is now a PoseStamped object
        # Extract translation (position)
        translation = np.array([
            vicon_data.pose.position.x,
            vicon_data.pose.position.y,
            vicon_data.pose.position.z
        ])
        # Extract rotation (orientation)
        rotation = Rotation.from_quat([
            vicon_data.pose.orientation.x,
            vicon_data.pose.orientation.y,
            vicon_data.pose.orientation.z,
            vicon_data.pose.orientation.w
        ])
        # Convert rotation to rotation matrix
        rotation_matrix = rotation.as_matrix()
        # Create the 4x4 transformation matrix
        transformation_matrix = np.eye(4)  # Initialize as identity matrix
        transformation_matrix[:3, :3] = rotation_matrix  # Set rotation part
        transformation_matrix[:3, 3] = -translation  # Set translation part
        pose  = torch.from_numpy(transformation_matrix.astype(np.float32))
    

    # Create a dummy batch
    batch = {
        'frame_id': step,
        'c2w': pose.unsqueeze(0),  
        'rgb': rgb_image.unsqueeze(0),  # [1, H, W, 3]
        'depth': depth_image.unsqueeze(0),  # [1, H, W]
        'direction': rays_d.unsqueeze(0)
    }

    return batch




def turtlebot_mapping(cfg, i):
    # num_frames 
    num_frames = cfg['data']['num_frames'] # how long you want to run it
    # preparing dataset info
    H, W = cfg['cam']['H'], cfg['cam']['W']
    fx, fy =  cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy =  cfg['cam']['cx'], cfg['cam']['cy']
    crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
    total_pixels = (H - crop_size*2) * (W - crop_size*2)
    num_rays_to_save = int(total_pixels * cfg['mapping']['n_pixels'])
    dataset_info = {'num_frames':num_frames, 'num_rays_to_save':num_rays_to_save, 'H':H, 'W':W }
    rays_d = get_camera_rays(H, W, fx, fy, cx, cy)
    max_depth = cfg['cam']['max_depth']

    # create mapping agent 
    agent_i = Mapping(cfg, i, dataset_info)
    agent_i.load_decoder(load_path=cfg['data']['load_path'])

    # create redis client 
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)

    # mapping
    for step in trange(num_frames):
        batch = data_loading(redis_client, max_depth, rays_d, step)
        agent_i.run(step, batch)




if __name__ == '__main__':

    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--agent_id', default=0, type=int, help='agent id')

    args = parser.parse_args()

    cfg = config.load_config(args.config)

    if cfg['multi_agents']['distributed_algorithm'] == 'AUQ_CADMM':
        cfg['multi_agents']['track_uncertainty'] = True


    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    turtlebot_mapping(cfg, args.agent_id)
