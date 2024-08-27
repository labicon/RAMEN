import argparse
import os
import glob
import time
import numpy as np
import torch
import cv2
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import DataLoader
from multiprocessing import Process, Queue
from queue import Empty

import config
from datasets.dataset import get_dataset
import sys 
import math
import matplotlib.pyplot as plt

def extrinsic_to_camera_params(extrinsic_matrix):
    # Extract rotation and translation
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]

    camera_position = -R.T @ t
    up_vector = R.T @ np.array([0, -1, 0])
    front_vector = R.T @ np.array([0, 0, -1])

    # LookAt is the point in the world the camera is looking at
    look_at_position = camera_position + front_vector

    return camera_position, look_at_position, up_vector


def normalize(x):
    return x / np.linalg.norm(x)


def create_camera_actor(i, color_list, is_gt=False, scale=0.005):
    cam_points = scale * np.array([
        [0,   0,   0],
        [-1,  -1, 1.5],
        [1,  -1, 1.5],
        [1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5]])

    cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4],
                          [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])
    points = []
    for cam_line in cam_lines:
        begin_points, end_points = cam_points[cam_line[0]
                                              ], cam_points[cam_line[1]]
        t_vals = np.linspace(0., 1., 100)
        begin_points, end_points
        point = begin_points[None, :] * \
            (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    color = (0.0, 0.0, 0.0) if is_gt else color_list[i]
    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor


def draw_trajectory(queue, output, cam_scale, estimate_c2w_list_agents, gt_c2w_list, num_frames, camera_params_extrinsic, bounding_box):

    draw_trajectory.queue = queue
    draw_trajectory.cameras = {}
    draw_trajectory.points = {}
    draw_trajectory.ix = 0
    draw_trajectory.warmup = 0
    draw_trajectory.mesh = None
    draw_trajectory.uncertainty_spheres = None
    draw_trajectory.frame_idx = 0
    draw_trajectory.traj_actor = None
    draw_trajectory.traj_actor_gt = None
    draw_trajectory.color_list =    [
                                        (1.0, 0.0, 0.0),   # Red
                                        (0.0, 1.0, 0.0),   # Green
                                        (0.0, 0.0, 1.0),   # Blue
                                        (1.0, 0.647, 0.0), # Orange
                                        (1.0, 1.0, 0.0),   # Yellow
                                        (0.502, 0.0, 0.502), # Purple
                                        (0.0, 1.0, 1.0),   # Cyan
                                        (1.0, 0.753, 0.796), # Pink
                                        (0.0, 0.502, 0.0), # Dark Green
                                        (1.0, 0.412, 0.706) # Hot Pink
                                    ]
    draw_trajectory.num_frames = num_frames
    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        while True:
            try:
                data = draw_trajectory.queue.get_nowait()
                if data[0] == 'pose':
                    i, pose, is_gt = data[1:]
                    if is_gt:
                        i += 100000

                    if i in draw_trajectory.cameras:
                        cam_actor, pose_prev = draw_trajectory.cameras[i]
                        pose_change = pose @ np.linalg.inv(pose_prev)

                        cam_actor.transform(pose_change)
                        vis.update_geometry(cam_actor)

                        if i in draw_trajectory.points:
                            pc = draw_trajectory.points[i]
                            pc.transform(pose_change)
                            vis.update_geometry(pc)

                    else:
                        cam_actor = create_camera_actor(i, draw_trajectory.color_list, is_gt, cam_scale)
                        cam_actor.transform(pose)
                        vis.add_geometry(cam_actor)

                    draw_trajectory.cameras[i] = (cam_actor, pose)

                elif data[0] == 'mesh':
                    meshfile = data[1]
                    if draw_trajectory.mesh is not None:
                        vis.remove_geometry(draw_trajectory.mesh)
                    draw_trajectory.mesh = o3d.io.read_triangle_mesh(meshfile)
                    draw_trajectory.mesh.compute_vertex_normals()
                    # flip face orientation
                    new_triangles = np.asarray(
                        draw_trajectory.mesh.triangles)[:, ::-1]
                    draw_trajectory.mesh.triangles = o3d.utility.Vector3iVector(
                        new_triangles)
                    draw_trajectory.mesh.triangle_normals = o3d.utility.Vector3dVector(
                        -np.asarray(draw_trajectory.mesh.triangle_normals))
                    vis.add_geometry(draw_trajectory.mesh)

                elif data[0] == 'uncertainty':
                    rgb = data[1]
                    vertices = data[2]
                    if draw_trajectory.uncertainty_spheres is not None:
                        vis.remove_geometry(draw_trajectory.uncertainty_spheres)
                    # draw_trajectory.uncertainty_pd = o3d.geometry.PointCloud()
                    # draw_trajectory.uncertainty_pd.points = o3d.utility.Vector3dVector(vertices)
                    # draw_trajectory.uncertainty_pd.colors = o3d.utility.Vector3dVector(rgb)
                    # vis.add_geometry(draw_trajectory.uncertainty_pd)

                    def create_sphere_mesh(radius, center, rgb):
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
                        sphere.translate(center)
                        sphere.paint_uniform_color(rgb)
                        return sphere
                    
                    def combine_meshes(meshes):
                        """Combine multiple meshes into one mesh."""
                        combined_mesh = o3d.geometry.TriangleMesh()
                        for mesh in meshes:
                            combined_mesh += mesh
                        return combined_mesh

                    radius = 0.025
                    spheres = [create_sphere_mesh(radius, vertices[i], rgb[i]) for i in range(rgb.shape[0])]
                    draw_trajectory.uncertainty_spheres = combine_meshes(spheres) # add one mesh to visualizer in one go is much faster than add these spheres one by one 
                    vis.add_geometry(draw_trajectory.uncertainty_spheres)

                elif data[0] == 'traj':
                    i, is_gt = data[1:]
                    if is_gt:
                        if draw_trajectory.traj_actor_gt is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor_gt)
                            # tmp = draw_trajectory.traj_actor_gt
                            # del tmp
                    else:
                        if draw_trajectory.traj_actor is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor)
                            # tmp = draw_trajectory.traj_actor
                            # del tmp

                    for agent_id, agent_est_c2w in enumerate(estimate_c2w_list_agents):
                        color = (0.0, 0.0, 0.0) if is_gt else draw_trajectory.color_list[agent_id]
                        traj_actor = o3d.geometry.PointCloud(
                            points=o3d.utility.Vector3dVector(gt_c2w_list[(draw_trajectory.num_frames*agent_id+1):(draw_trajectory.num_frames*agent_id+i), :3, 3] if is_gt else agent_est_c2w[1:i, :3, 3]))
                        traj_actor.paint_uniform_color(color)

                        if is_gt:
                            draw_trajectory.traj_actor_gt = traj_actor
                            vis.add_geometry(draw_trajectory.traj_actor_gt)
                        else:
                            draw_trajectory.traj_actor = traj_actor
                            vis.add_geometry(draw_trajectory.traj_actor)

                elif data[0] == 'reset':
                    draw_trajectory.warmup = -1

                    for i in draw_trajectory.points:
                        vis.remove_geometry(draw_trajectory.points[i])

                    for i in draw_trajectory.cameras:
                        vis.remove_geometry(draw_trajectory.cameras[i][0])

                    draw_trajectory.cameras = {}
                    draw_trajectory.points = {}

            except Empty:
                break

        # hack to allow interacting with vizualization during inference
        if len(draw_trajectory.cameras) >= draw_trajectory.warmup:
            cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()

    vis = o3d.visualization.Visualizer()

    vis.register_animation_callback(animation_callback)
    vis.create_window(window_name=output, height=1080, width=1920)
    vis.get_render_option().point_size = 4
    vis.get_render_option().mesh_show_back_face = False
    vis.get_render_option().show_coordinate_frame = True #red-x, green-y, blue-z 

    # add bounding box 
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounding_box[:, 0], max_bound=bounding_box[:, 1])
    bbox.color = (1, 0, 0)  # Red color
    vis.add_geometry(bbox)

    # set up view control 
    ctr = vis.get_view_control()
    if camera_params_extrinsic is not None:
        camera_position, look_at_position, up_vector = extrinsic_to_camera_params(camera_params_extrinsic)
        # Set the camera parameters
        ctr.set_front((look_at_position - camera_position))
        ctr.set_lookat(look_at_position)
        ctr.set_up(up_vector)
        ctr.set_zoom(1.0)  # Adjust zoom as necessary

    vis.run()

    # get current camera parameters 
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    np.save('camera_params_extrinsic.npy', camera_params.extrinsic)
    print('camera parameters saved for view control')

    vis.destroy_window()


class SLAMFrontend:

    def __init__(self, output, cam_scale=1,
                 estimate_c2w_list_agents=None, gt_c2w_list=None, num_frames=0, camera_params_extrinsic=None, bounding_box=None):
        self.queue = Queue()
        self.p = Process(target=draw_trajectory, args=(
            self.queue, output, cam_scale, 
            estimate_c2w_list_agents, gt_c2w_list, num_frames, camera_params_extrinsic, bounding_box))

    def update_pose(self, index, pose, gt=False):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()

        pose[:3, 2] *= -1
        self.queue.put_nowait(('pose', index, pose, gt))
        
    def update_mesh(self, path):
        self.queue.put_nowait(('mesh', path))

    def update_uncertainty(self, rgb, vertices):
        self.queue.put_nowait(('uncertainty', rgb, vertices))

    def update_cam_trajectory(self, c2w_list, gt):
        self.queue.put_nowait(('traj', c2w_list, gt))

    def reset(self):
        self.queue.put_nowait(('reset', ))

    def start(self):
        self.p.start()
        return self

    def join(self):
        self.p.join()


def get_est_c2w(ckptsdir):
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'checkpoint' in f] 
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)

            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = list(ckpt['pose'].values())
    estimate_c2w_list = torch.stack(estimate_c2w_list).cpu().numpy()
    num_frames = len(estimate_c2w_list)
    return estimate_c2w_list, num_frames


def get_grid_resolution(cfg):
    bounding_box = np.asarray(cfg['mapping']['bound'])
    dim_max = (bounding_box[:,1] - bounding_box[:,0]).max()
    N_max = int(dim_max / cfg['grid']['voxel_sdf'])

    F = 2 
    d = 3 
    T = 2**cfg['grid']['hash_size']
    N_min = 16 
    L = 16
    b = np.exp2(np.log2(N_max  / N_min) / (L - 1))

    def next_multiple(val, divisor):
        div_round_up = (val+divisor-1) // divisor 
        return div_round_up * divisor

    params_in_level_list = []
    N_l_list = []
    for l in range(L):
        N_l = math.ceil(b**l * N_min - 1) + 1 # this is different from how N_l is calculated in the paper
        N_l_list.append(N_l)

        params_in_level = N_l**d
        params_in_level = next_multiple(params_in_level, 8) # to make sure memory accesses will be aligned, this will lead to non-integer cube root 
        params_in_level = min(params_in_level, T) 
        params_in_level_list.append(params_in_level*F)

    return N_l_list[0], params_in_level_list[0]


def process_uncertainty_file(uncertaintyFile, cfg, N_l, params_in_level):
    """
        @return : rgb, vertices
    """
    # get color 
    uncertainty_tensor = torch.load(uncertaintyFile)[:params_in_level]
    uncertainty_tensor = uncertainty_tensor.view(-1,2).sum(-1) 
    uncertainty_tensor /= torch.max(uncertainty_tensor) # normalize
    rgb = plt.cm.cool(uncertainty_tensor.cpu().numpy())[:,:3]

    # get grid 
    bbox = np.asarray(cfg['mapping']['bound'])
    x = np.linspace( bbox[0,0], bbox[0,1], num=N_l)
    y = np.linspace( bbox[1,0], bbox[1,1], num=N_l)
    z = np.linspace( bbox[2,0], bbox[2,1], num=N_l)
    grid = np.meshgrid(z, y, x, indexing='ij')
    vertices = np.stack(grid, axis=-1).reshape(-1, 3)[:,::-1] # so that the 2nd dimension order is (x,y,z)

    return rgb, vertices


if __name__ == '__main__':
    """
        Black: ground truth 
        python -W ignore .\visualizer_agents.py --config .\configs\Replica\office0_agents.yaml --agent 1
        -W ignore for ignoring warning
    """
    parser = argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--vis_input_frame',
                        action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj',
                        action='store_true', help='not visualize gt trajectory')
    parser.add_argument('--agent', default=0, type=int, help='which agent mesh to show')  
    parser.add_argument('--show_last',
                        action='store_true', help='show the whole trajectories and the last mesh')
    parser.add_argument('--mesh_only',
                        action='store_true', help='only show mesh')
    parser.add_argument('--show_uncertainty',
                        action='store_true', help='visualize grid uncertainty')
    args = parser.parse_args()
    cfg = config.load_config(args.config)

    if os.path.exists('camera_params_extrinsic.npy'):
        print('Get camera parameters for view control')
        camera_params_extrinsic = np.load('camera_params_extrinsic.npy')
    else:
        camera_params_extrinsic = None

    # get estimated poses
    ckptsdir_list = glob.glob(os.path.join(cfg['data']['output'], cfg['data']['exp_name'], 'agent_*'))
    ckptsdir_list = sorted(ckptsdir_list, key=lambda x: int(x.split('_')[-1]))
    estimate_c2w_list_agents = []
    for dir in ckptsdir_list:
        estimate_c2w_list, num_frames = get_est_c2w(dir)
        estimate_c2w_list_agents.append(estimate_c2w_list)

    # get gt poses
    dataset = get_dataset(cfg)
    gt_c2w_list = dataset.poses
    gt_c2w_list = torch.stack(gt_c2w_list).cpu().numpy()
    frontend = SLAMFrontend(cfg['data']['exp_name'], cam_scale=0.3,
                            estimate_c2w_list_agents=estimate_c2w_list_agents, gt_c2w_list=gt_c2w_list, 
                            num_frames=num_frames, camera_params_extrinsic=camera_params_extrinsic, bounding_box=np.asarray(cfg['mapping']['bound'])).start()
    
    # prepare for uncertainty visualization 
    if args.show_uncertainty:
        N_l, params_in_level = get_grid_resolution(cfg) #TODO: for now we only visualize level 0 grid
        print(f'N_l = {N_l}, params_in_level = {params_in_level}')

    start_frame = num_frames - 1 if args.show_last else 0
    for i in tqdm(range(start_frame, num_frames)): # tqdm progress bar starts with 1
        # show every fourth frame for speed up
        if args.vis_input_frame and i % 4 == 0:
            for agent_id in range(len(estimate_c2w_list_agents)):
                ret = dataset[agent_id*num_frames + i]
                gt_color = ret['rgb']
                gt_depth = ret['depth']
                depth_np = gt_depth.numpy()
                color_np = (gt_color.numpy()*255).astype(np.uint8)
                depth_np = depth_np / np.max(depth_np) * 255
                depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
                depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
                color_np = np.clip(color_np, 0, 255)
                whole = np.concatenate([color_np, depth_np], axis=0)
                H, W, _ = whole.shape
                whole = cv2.resize(whole, (W//4, H//4))
                # Use the agent_id to create unique window names
                window_name = f'Agent {agent_id} Input RGB-D Sequence'
                # Display the image in a separate window for each agent
                cv2.imshow(window_name, whole[:, :, ::-1])
            cv2.waitKey(1)
        time.sleep(0.03) # don't delete this, otherwise loop will immediately ends before mesh and trajectories can be updated

        meshfile = f'{ckptsdir_list[args.agent]}/mesh_track{i}.ply'
        if os.path.isfile(meshfile):
            frontend.update_mesh(meshfile)
        
        if args.show_uncertainty:
            uncertaintyFile = f'{ckptsdir_list[args.agent]}/uncertain_track{i}.pt'
            if os.path.isfile(uncertaintyFile):
                rgb, vertices = process_uncertainty_file(uncertaintyFile, cfg, N_l, params_in_level)
                frontend.update_uncertainty(rgb, vertices)

        if args.mesh_only == False:
            for id in range(len(estimate_c2w_list_agents)):
                frontend.update_pose(id, estimate_c2w_list_agents[id][i], gt=False)
                if not args.no_gt_traj:
                    frontend.update_pose(id, gt_c2w_list[id*num_frames+i], gt=True)
            # the visualizer might get stucked if update every frame
            # with a long sequence (10000+ frames)
            if (i+1) % 10 == 0 or (i+1) == num_frames:
                frontend.update_cam_trajectory(i, gt=False)
                if not args.no_gt_traj:
                    frontend.update_cam_trajectory(i, gt=True)

