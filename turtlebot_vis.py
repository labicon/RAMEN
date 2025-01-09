import open3d as o3d
import numpy as np
import os
import time
import glob
import argparse
import config # coslam load config 
import torch

def get_latest_mesh(directory):
    list_of_files = glob.glob(os.path.join(directory, 'mesh_track*.ply'))
    if not list_of_files:
        return None

    latest_file = max(list_of_files, key=lambda f: int(f.split('mesh_track')[-1].split('.ply')[0]))
    return latest_file




def get_latest_ckpt(directory):
    list_of_files = glob.glob(os.path.join(directory, 'checkpoint*.pt'))
    if not list_of_files:
        return None

    latest_file = max(list_of_files, key=lambda f: int(f.split('checkpoint_')[-1].split('.pt')[0]))
    return latest_file



def create_camera_actor(scale=0.1):
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
    color = (238 / 255.,130 / 255.,238 / 255.) # violet
    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor




def visualize_ply_dynamic(directory, cfg):
    try:
        vis = o3d.visualization.Visualizer()
        ctr = vis.get_view_control()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True

        # add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # add  bounding box 
        bounding_box = np.asarray(cfg['mapping']['bound'])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounding_box[:, 0], max_bound=bounding_box[:, 1])
        bbox.color = (1, 0, 0)  # Red color
        vis.add_geometry(bbox)

        current_mesh_path = None
        current_ckpt_path = None
        mesh = None  # Initialize mesh outside the loop
        view_params = None
        cam_actor = None
        while True:
            latest_mesh_path = get_latest_mesh(directory)
            latest_ckpt_path = get_latest_ckpt(directory)

            # update camera actor 
            if latest_ckpt_path != current_ckpt_path:
                current_ckpt_path = latest_ckpt_path
                if latest_ckpt_path is not None:
                    time.sleep(1)
                    print(f"latest ckpt = {latest_ckpt_path}")
                    ckpt = torch.load(latest_ckpt_path, map_location=torch.device('cpu'))
                    estimate_c2w_list = list(ckpt['pose'].values())
                    estimate_c2w_list = torch.stack(estimate_c2w_list).cpu().numpy()

                    pose = estimate_c2w_list[-1]
                    pose[:3, 2] *= -1 # follow visualizer.py
                    if cam_actor == None:
                        cam_actor = create_camera_actor()
                        cam_actor.transform(pose) # rotation from body to world 
                        vis.add_geometry(cam_actor)
                        pose_prev = pose
                    else:
                        pose_change = pose @ np.linalg.inv(pose_prev)
                        cam_actor.transform(pose_change)
                        vis.update_geometry(cam_actor)
                        pose_prev = pose

            
            if latest_mesh_path != current_mesh_path:
                current_mesh_path = latest_mesh_path

                if mesh is not None:
                    vis.remove_geometry(mesh)

                if latest_mesh_path is not None:
                    time.sleep(1) # need to wait a bit for the file to be fully generated
                    print(f"latest mesh = {latest_mesh_path}")
                    mesh = o3d.io.read_triangle_mesh(latest_mesh_path)
                    mesh.compute_vertex_normals()
                    vis.add_geometry(mesh)
                    if view_params is not None:
                        ctr = vis.get_view_control()
                        ctr.convert_from_pinhole_camera_parameters(view_params, allow_arbitrary=True)

            # render options


            vis.poll_events()
            vis.update_renderer()
            ctr = vis.get_view_control()
            view_params = ctr.convert_to_pinhole_camera_parameters()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        vis.destroy_window()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--config', default='configs/turtlebot/test.yaml', type=str, help='Path to config file.')
    parser.add_argument('--directory', default='output/turtlebot/test/agent_0', type=str, help='directory to find mesh files')
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    
    visualize_ply_dynamic(args.directory, cfg)