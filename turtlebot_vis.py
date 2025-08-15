import open3d as o3d
import numpy as np
import os
import time
import glob
import argparse
import config # coslam load config 
import torch
from visualizer import get_grid_resolution, process_uncertainty_file

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


def get_latest_uncertainty(directory):
    list_of_files = glob.glob(os.path.join(directory, 'uncertain_track*.pt'))
    if not list_of_files:
        return None

    latest_file = max(list_of_files, key=lambda f: int(f.split('uncertain_track')[-1].split('.pt')[0]))
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




def visualize_ply_dynamic(directory, show_uncertainty, cfg):
    try:

        vis = o3d.visualization.Visualizer()
        ctr = vis.get_view_control()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True

        # ready for uncertainty visualization 
        if args.show_uncertainty: 
            N_l, params_in_level = get_grid_resolution(cfg) #TODO: for now we only visualize level 0 grid
            print(f'N_l = {N_l}, params_in_level = {params_in_level}')
            
        # add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # add  bounding box 
        bounding_box = np.asarray(cfg['mapping']['bound'])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounding_box[:, 0], max_bound=bounding_box[:, 1])
        bbox.color = (1, 0, 0)  # Red color
        vis.add_geometry(bbox)

        wait_time = 0.5 # in case read before a file is ready
        current_mesh_path = None
        current_ckpt_path = None
        current_uncertainty_path = None 
        mesh = None  # Initialize mesh outside the loop
        uncertainty_spheres = None
        view_params = None
        cam_actor = None
        while True:
            latest_mesh_path = get_latest_mesh(directory)
            latest_ckpt_path = get_latest_ckpt(directory)
            if args.show_uncertainty: 
                latest_uncertainty_path = get_latest_uncertainty(directory)
            else:
                latest_uncertainty_path = None

            # update camera actor 
            if latest_ckpt_path != current_ckpt_path:
                current_ckpt_path = latest_ckpt_path
                if latest_ckpt_path is not None:
                    time.sleep(wait_time)
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


            # update uncertainty
            if latest_uncertainty_path != current_uncertainty_path:
                current_uncertainty_path = latest_uncertainty_path

                if uncertainty_spheres is not None:
                    vis.remove_geometry(uncertainty_spheres)
                    
                if latest_uncertainty_path is not None:
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
                    time.sleep(wait_time)
                    rgb, vertices = process_uncertainty_file(latest_uncertainty_path, cfg, N_l, params_in_level, vis_type='uncertainty')
                    spheres = [create_sphere_mesh(radius, vertices[i], rgb[i]) for i in range(rgb.shape[0])]
                    uncertainty_spheres = combine_meshes(spheres) # add one mesh to visualizer in one go is much faster than add these spheres one by one 
                    vis.add_geometry(uncertainty_spheres)




            # update mesh 
            if latest_mesh_path != current_mesh_path:
                current_mesh_path = latest_mesh_path

                if mesh is not None:
                    vis.remove_geometry(mesh)

                if latest_mesh_path is not None:
                    time.sleep(wait_time) # need to wait a bit for the file to be fully generated
                    print(f"latest mesh = {latest_mesh_path}")
                    mesh = o3d.io.read_triangle_mesh(latest_mesh_path)
                    mesh.compute_vertex_normals()

                    # flip face orientation. can't work on my laptop somehow
                    new_triangles = np.asarray(mesh.triangles)[:, ::-1]
                    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
                    mesh.triangle_normals = o3d.utility.Vector3dVector(-np.asarray(mesh.triangle_normals))
                
                    vis.add_geometry(mesh)
            

            # view control
            if view_params is not None:
                ctr = vis.get_view_control()
                ctr.convert_from_pinhole_camera_parameters(view_params, allow_arbitrary=True)


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
    parser.add_argument('--show_uncertainty',
                        action='store_true', help='visualize grid uncertainty')
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    directory = os.path.join(cfg['data']['output'], cfg['data']['exp_name'], 'agent_0')
    print(f'Save path is {directory}')
    visualize_ply_dynamic(directory, args.show_uncertainty, cfg)