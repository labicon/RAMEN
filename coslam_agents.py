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
import shutil 

from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import networkx as nx 
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector as p2v

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion

import sys


class CoSLAM():
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
        self.create_optimizer()

        self.dist_algorithm = config['multi_agents']['distributed_algorithm']
        self.track_uncertainty = config['multi_agents']['track_uncertainty']
        if self.track_uncertainty:
            self.uncertainty_tensor = torch.zeros(self.model.embed_fn.params.size()).to(self.device)

        # initialize dual variable 
        theta_i = p2v(self.model.parameters())
        self.p_i = torch.zeros(theta_i.size()).to(self.device)
        # a list to hold neighbor model parameters, and uncertainty tensor (optional)
        self.neighbors = []
        # step size in the gradient ascent of the dual variable
        self.rho = config['multi_agents']['rho']
        
        self.com_perIter = 0 # communication cost in MB per communication iteration 
        self.com_total = 0 # total accumulated communication cost in MB 

        self.gt_pose = config['tracking']['gt_pose']
        print(f'If agent{self.agent_id} uses gt pose: {self.gt_pose}')


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
    

    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
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
    

    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                               {"params": cur_trans, "lr": self.config[task]['lr_trans']}])
        
        return cur_rot, cur_trans, pose_optimizer
 

    def communicate(self,input):
        model_j = input[0]
        theta_j = p2v(model_j.parameters()).detach()

        if self.dist_algorithm == 'AUQ_CADMM':
            uncertainty_j = input[1].detach()
            self.neighbors.append( [theta_j, uncertainty_j] )
        elif self.dist_algorithm == 'CADMM':
            self.neighbors.append( [theta_j] )


    def dual_update(self, theta_i_k):
        for neighbor in self.neighbors:
            theta_j_k = neighbor[0]
            self.p_i += self.rho * (theta_i_k - theta_j_k)    


    def dual_update_AUQ_CADMM(self, theta_i_k):
        """
            @return rho_matrices:
        """
        rho_matrices = []
        uncertainty_i = self.uncertainty_tensor
        for neighbor in self.neighbors:
            theta_j_k = neighbor[0]
            uncertainty_j = neighbor[1]
            Rho_ij_diag = torch.div( torch.exp(uncertainty_j), torch.exp(uncertainty_i) + torch.exp(uncertainty_j) )
            Rho_ij = torch.diag(Rho_ij_diag) * self.rho
            rho_matrices.append(Rho_ij)
            self.p_i +=  Rho_ij @ (theta_i_k - theta_j_k)    

        return rho_matrices


    def primal_update(self, theta_i_k, loss):
        theta_i = p2v(self.model.parameters())
        loss = loss + torch.dot(theta_i, self.p_i)
        for neighbor in self.neighbors:
            theta_j_k = neighbor[0]
            difference = self.rho * torch.norm(theta_i - (theta_i_k+theta_j_k)/2)**2
            loss += difference
        return loss
    

    def primal_update_AUQ_CADMM(self, theta_i_k, loss, rho_matrices):
        theta_i = p2v(self.model.parameters())
        loss = loss + torch.dot(theta_i, self.p_i)
        for neighbor, Rho_ij in zip(self.neighbors, rho_matrices):
            theta_j_k = neighbor[0]            
            difference = theta_i - (theta_i_k+theta_j_k)/2
            weighted_norm = difference @ Rho_ij @ difference
            loss += weighted_norm
        return loss


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
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])

        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
            pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
            poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
        
        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1]) 

        theta_i_k = p2v(self.model.parameters()).detach()
        if dist_algorithm == 'CADMM':
            self.dual_update(theta_i_k)
        elif dist_algorithm == 'AUQ_CADMM':
            rho_matrices = self.dual_update_AUQ_CADMM(theta_i_k)

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
            if self.track_uncertainty:
                grid_grad = self.model.embed_fn.params.grad
                grid_has_grad = (torch.abs(grid_grad) > 0).to(torch.int32)
                self.uncertainty_tensor += grid_has_grad

            if dist_algorithm == 'CADMM':
                loss = self.primal_update(theta_i_k, loss)
                loss.backward(retain_graph=True)
            elif dist_algorithm == 'AUQ_CADMM':
                loss = self.primal_update_AUQ_CADMM(theta_i_k, loss, rho_matrices)
                loss.backward(retain_graph=True)

            self.map_optimizer.step()
            
            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # zero_grad here
                pose_optimizer.zero_grad()
        
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
 

    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev
            
        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id-2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            delta = c2w_est_prev@c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta@c2w_est_prev
        
        return self.est_c2w_data[frame_id]


    def tracking_render(self, batch, frame_id):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)

        # Initialize current pose
        if self.config['tracking']['iter_point'] > 0:
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        indice = None
        best_sdf_loss = None
        thresh=0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)

        # Start tracking
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

            # Note here we fix the sampled points for optimisation
            if indice is None:
                indice = self.select_samples(self.dataset_info['H']-iH*2, self.dataset_info['W']-iW*2, self.config['tracking']['sample'])
            
                # Slicing
                indice_h, indice_w = indice % (self.dataset_info['H'] - iH * 2), indice // (self.dataset_info['H'] - iH * 2)
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w_est[...,:3, -1].repeat(self.config['tracking']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            
            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh +=1
            
            if thresh >self.config['tracking']['wait_iters']:
                break

            loss.backward()
            pose_optimizer.step()
        
        if self.config['tracking']['best']:
            # Use the pose with smallest loss
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        if self.gt_pose:
            self.est_c2w_data[frame_id] = c2w_gt

       # Save relative pose of non-keyframes
        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
        
        print('\nAgent{}, Best loss: {}, Last loss{}'.format(self.agent_id, F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
    

    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i] 
                poses[i] = delta @ c2w_key
        
        return poses


    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
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
            print(f'\nAgent {self.agent_id} add keyframe:{i}')

        if i % self.config['mesh']['vis']==0:
            self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])


        if i == (self.dataset_info['num_frames']-1):
            model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'checkpoint{}.pt'.format(i)) 
            self.save_ckpt(model_savepath)
            self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])
        

def create_agent_graph(cfg, dataset):
    """
        @param cfg:
        @param dataset:
        @return G: created graph
        @return frames_per_agent:
    """
    G = nx.Graph()
    node_list = []

    num_agents = cfg['multi_agents']['num_agents']
    frames_per_agent = len(dataset) // num_agents
    dataset_info = {'num_frames':frames_per_agent, 'num_rays_to_save':dataset.num_rays_to_save, 'H':dataset.H, 'W':dataset.W }

    for i in range(num_agents):
        print(f'\nCreating agnet {i}')
        agent_i = CoSLAM(cfg, i, dataset_info)
        node_list.append( [ i, {"agent": agent_i} ] )

    G.add_nodes_from(node_list) 
    G.add_edges_from(cfg['multi_agents']['edges_list'])

    # plot graph
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()  

    return G, frames_per_agent


def get_data_memory(dataset, cfg, frames_per_agent):
    num_agents = cfg['multi_agents']['num_agents']
    output_path = os.path.join(cfg['data']['output'], cfg['data']['exp_name'])
    rgb = dataset[0]['rgb']
    depth = dataset[0]['depth']
    rgb_memory = torch.numel(rgb)*rgb.element_size() / 1e6 # bytes to megabytes
    depth_memory = torch.numel(depth)*depth.element_size() / 1e6 # bytes to megabytes
    single_size= f'size of a rgb img and a depth img: {rgb_memory + depth_memory} MB\n'
    total_size = f'total size of all images shared for centralized training: {(rgb_memory + depth_memory)*frames_per_agent*(num_agents-1)} MB\n'
    
    # Save to a text file
    print("Save Memory Info")
    with open(os.path.join(output_path, 'memory_sizes.txt'), 'w') as file:
        file.write(single_size)
        file.write(total_size)


def get_model_memory(model):
    model_tensor = p2v(model.parameters())
    model_size = torch.numel(model_tensor)*model_tensor.element_size() / 1e6 # bytes to megabytes
    return model_size


def train_multi_agent(cfg):

    dataset = get_dataset(cfg)

    G, frames_per_agent = create_agent_graph(cfg, dataset)

    get_data_memory(dataset, cfg, frames_per_agent)

    for step in trange(0, frames_per_agent, smoothing=0):
        # commnuication
        if step % cfg['multi_agents']['com_every'] == 0:
            for i, nbrs in G.adj.items():
                print(f'\nAgent {i} Communicating')
                agent_i = G.nodes[i]['agent']
                agent_i.neighbors = [] # clear communication buffer, only save the latest weights
                agent_i.com_perIter = 0
                for j, edge_attr in nbrs.items():
                    agent_j = G.nodes[j]['agent']
                    if cfg['multi_agents']['distributed_algorithm'] == 'AUQ_CADMM':
                        agent_i.communicate([agent_j.model, agent_j.uncertainty_tensor])
                        model_size = get_model_memory(agent_j.model) # + get_model_memory(agent_j.uncertainty_tensor) #TODO: fix communication
                    elif cfg['multi_agents']['distributed_algorithm'] == 'CADMM':
                        agent_i.communicate([agent_j.model])
                        model_size = get_model_memory(agent_j.model)
                    agent_i.com_perIter += model_size
                    agent_i.com_total += model_size
             
        # update
        for i, nbrs in G.adj.items():
            agent_i = G.nodes[i]['agent']
            batch_i = dataset[i*frames_per_agent+step] 
            batch_i["frame_id"] = step
            for key in list(batch_i.keys())[1:]:
                batch_i[key] = batch_i[key].unsqueeze(0)
            agent_i.run(step, batch_i)

    # write communication info
    output_path = os.path.join(cfg['data']['output'], cfg['data']['exp_name'])
    for i, nbrs in G.adj.items():
        agent_i = G.nodes[i]['agent']
        com_perIter = f'Agent {i} message received per communication iteration: {agent_i.com_perIter} MB\n'
        com_total = f'Agent {i} total message received: {agent_i.com_total} MB\n'
        with open(os.path.join(output_path, 'memory_sizes.txt'), 'a') as file: # mode 'a' for append mode, so you can add new content without deleting the previous one
            file.write(com_perIter)
            file.write(com_total)
    print("Agent Communication Info Saved")


if __name__ == '__main__':
    """
         python .\coslam_agents.py --config .\configs\Azure\apartment_agents.yaml
    """

    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    if cfg['multi_agents']['distributed_algorithm'] == 'AUQ_CADMM':
        cfg['multi_agents']['track_uncertainty'] = True


    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))


    # multi-agent training 
    train_multi_agent(cfg)
