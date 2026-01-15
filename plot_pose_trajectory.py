import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_and_process_poses(path, sc_factor=1.0):
    """
    Loads and processes poses from a trajectory file.
    The processing is based on the ReplicaDataset in datasets/dataset.py.
    """
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w[:3, 3] *= sc_factor
        poses.append(c2w)
    return poses

def main():
    """
    Main function to load data, and generate plots.
    """
    traj_file = 'data/Replica/oogway_realsense_Jan15/traj.txt'
    # sc_factor is from config['data']['sc_factor'], assuming 1.0 as default
    poses = load_and_process_poses(traj_file)

    positions = []
    euler_angles = []

    for pose in poses:
        # Extract position
        positions.append(pose[:3, 3])

        # Extract rotation matrix
        rotation_matrix = pose[:3, :3]
        
        # Convert rotation matrix to Euler angles
        r = R.from_matrix(rotation_matrix)
        euler = r.as_euler('xyz', degrees=True)
        euler_angles.append(euler)

    positions = np.array(positions)
    euler_angles = np.array(euler_angles)

    # Plot 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Robot Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot 3D Trajectory')
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=20., azim=-35)
    plt.savefig('trajectory_3d.png')

    # Plot positions
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10,8))
    axs[0].plot(positions[:, 0], label='X')
    axs[0].set_ylabel('meters')
    axs[0].set_title('Position')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(positions[:, 1], label='Y')
    axs[1].set_ylabel('meters')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(positions[:, 2], label='Z')
    axs[2].set_ylabel('meters')
    axs[2].set_xlabel('Frame')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('position.png')

    # Plot Euler angles
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10,8))
    axs[0].plot(euler_angles[:, 0], label='X')
    axs[0].set_ylabel('degrees')
    axs[0].set_title('Euler Angles')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(euler_angles[:, 1], label='Y')
    axs[1].set_ylabel('degrees')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(euler_angles[:, 2], label='Z')
    axs[2].set_ylabel('degrees')
    axs[2].set_xlabel('Frame')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('rotation.png')

if __name__ == '__main__':
    main()
