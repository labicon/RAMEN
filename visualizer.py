import argparse
import os
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





def normalize(x):
    return x / np.linalg.norm(x)


def create_camera_actor(i, is_gt=False, scale=0.005):
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
    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor


def draw_trajectory(queue, output, init_pose, cam_scale, near, estimate_c2w_list, gt_c2w_list):

    draw_trajectory.queue = queue
    draw_trajectory.cameras = {}
    draw_trajectory.points = {}
    draw_trajectory.ix = 0
    draw_trajectory.warmup = 0
    draw_trajectory.mesh = None
    draw_trajectory.frame_idx = 0
    draw_trajectory.traj_actor = None
    draw_trajectory.traj_actor_gt = None

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
                        cam_actor = create_camera_actor(i, is_gt, cam_scale)
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


                elif data[0] == 'traj':
                    i, is_gt = data[1:]

                    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
                    traj_actor = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(gt_c2w_list[1:i, :3, 3] if is_gt else estimate_c2w_list[1:i, :3, 3]))
                    traj_actor.paint_uniform_color(color)

                    if is_gt:
                        if draw_trajectory.traj_actor_gt is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor_gt)
                            tmp = draw_trajectory.traj_actor_gt
                            del tmp
                        draw_trajectory.traj_actor_gt = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor_gt)
                    else:
                        if draw_trajectory.traj_actor is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor)
                            tmp = draw_trajectory.traj_actor
                            del tmp
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

    ctr = vis.get_view_control()
    ctr.set_constant_z_near(near)
    ctr.set_constant_z_far(1000)

    # set he viewer's pose in the back of the first frame's pose
    param = ctr.convert_to_pinhole_camera_parameters()
    init_pose[:3, 3] += 2*normalize(init_pose[:3, 2])
    init_pose[:3, 2] *= -1
    init_pose[:3, 1] *= -1
    init_pose = np.linalg.inv(init_pose)

    param.extrinsic = init_pose 
    ctr.convert_from_pinhole_camera_parameters(param) 

    vis.run()
    vis.destroy_window()


class SLAMFrontend:
    def __init__(self, output, init_pose, cam_scale=1,
                 near=0, estimate_c2w_list=None, gt_c2w_list=None):
        self.queue = Queue()
        self.p = Process(target=draw_trajectory, args=(
            self.queue, output, init_pose, cam_scale, 
            near, estimate_c2w_list, gt_c2w_list))

    def update_pose(self, index, pose, gt=False):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()

        pose[:3, 2] *= -1
        self.queue.put_nowait(('pose', index, pose, gt))
        
    def update_mesh(self, path):
        self.queue.put_nowait(('mesh', path))

    def update_cam_trajectory(self, c2w_list, gt):
        self.queue.put_nowait(('traj', c2w_list, gt))

    def reset(self):
        self.queue.put_nowait(('reset', ))

    def start(self):
        self.p.start()
        return self

    def join(self):
        self.p.join()




if __name__ == '__main__':
    """
        Black: ground truth 
        Red: Predicted trajectory
        python .\visualizer.py --config .\configs\Replica\office0_agents.yaml --agent agent_2 --start_frame 1332
    """
    parser = argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--vis_input_frame',
                        action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj',
                        action='store_true', help='not visualize gt trajectory')
    parser.add_argument('--agent', default=None, type=str)
    parser.add_argument('--start_frame', default=0, type=int)
    args = parser.parse_args()
    cfg = config.load_config(args.config)

    # get estimated poses
    if args.agent is not None:
        ckptsdir = os.path.join(cfg['data']['output'], cfg['data']['exp_name'], args.agent) 
    else:
        ckptsdir = os.path.join(cfg['data']['output'], cfg['data']['exp_name']) 
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'pt' in f] 
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)

            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = list(ckpt['pose'].values())
    estimate_c2w_list = torch.stack(estimate_c2w_list).cpu().numpy()
    num_frames = len(estimate_c2w_list)

    # get gt poses
    dataset = get_dataset(cfg)
    gt_c2w_list = dataset.poses
    gt_c2w_list = torch.stack(gt_c2w_list).cpu().numpy()[args.start_frame : args.start_frame+num_frames]
    frontend = SLAMFrontend(ckptsdir, init_pose=estimate_c2w_list[0], cam_scale=0.3,
                            near=0, estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list).start()


    for i in tqdm(range(0, len(estimate_c2w_list))):
        # show every second frame for speed up
        if args.vis_input_frame and i % 2 == 0:
            ret = dataset[args.start_frame + i]
            gt_color = ret['rgb']
            gt_depth = ret['depth']
            depth_np = gt_depth.numpy()
            color_np = (gt_color.numpy()*255).astype(np.uint8)
            depth_np = depth_np/np.max(depth_np)*255
            depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
            depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
            color_np = np.clip(color_np, 0, 255)
            whole = np.concatenate([color_np, depth_np], axis=0)
            H, W, _ = whole.shape
            whole = cv2.resize(whole, (W//4, H//4))
            cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])
            cv2.waitKey(1)
        time.sleep(0.03)
        meshfile = f'{ckptsdir}/mesh_track{i}.ply'
        if os.path.isfile(meshfile):
            frontend.update_mesh(meshfile)
        frontend.update_pose(1, estimate_c2w_list[i], gt=False)
        if not args.no_gt_traj:
            frontend.update_pose(1, gt_c2w_list[i], gt=True)
        # the visualizer might get stucked if update every frame
        # with a long sequence (10000+ frames)
        if i % 10 == 0:
            frontend.update_cam_trajectory(i, gt=False)
            if not args.no_gt_traj:
                frontend.update_cam_trajectory(i, gt=True)

