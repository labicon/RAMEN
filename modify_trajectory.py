import numpy as np
from scipy.spatial.transform import Rotation as R

def load_poses_raw(path):
    """
    Loads poses from a trajectory file without any processing.
    """
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        poses.append(c2w)
    return poses

def process_pose(c2w, sc_factor=1.0):
    """
    Processes a pose matrix like in ReplicaDataset.
    """
    c2w_processed = c2w.copy()
    c2w_processed[:3, 1] *= -1
    c2w_processed[:3, 2] *= -1
    c2w_processed[:3, 3] *= sc_factor
    return c2w_processed

def deprocess_pose(c2w, sc_factor=1.0):
    """
    Reverses the processing of a pose matrix.
    """
    c2w_deprocessed = c2w.copy()
    # Invert the scaling
    c2w_deprocessed[:3, 3] /= sc_factor
    # Invert the sign changes
    c2w_deprocessed[:3, 1] *= -1
    c2w_deprocessed[:3, 2] *= -1
    return c2w_deprocessed

def save_poses(path, poses):
    """
    Saves poses to a trajectory file.
    """
    with open(path, "w") as f:
        for pose in poses:
            f.write(" ".join(map(str, pose.flatten())) + "\n")

def main():
    traj_file = 'data/Replica/oogway_realsense_Jan15/traj.txt'
    raw_poses = load_poses_raw(traj_file)
    
    if not raw_poses:
        print("No poses found in the file.")
        return

    # Process poses to get into the right coordinate system for euler angles
    processed_poses = [process_pose(p) for p in raw_poses]

    # Get initial values from the first processed pose
    first_pose = processed_poses[0]
    initial_position = first_pose[:3, 3]
    initial_z = initial_position[2]
    
    initial_rotation_matrix = first_pose[:3, :3]
    initial_euler = R.from_matrix(initial_rotation_matrix).as_euler('xyz', degrees=True)
    initial_roll = initial_euler[0]
    initial_pitch = initial_euler[1]

    modified_processed_poses = []

    for pose in processed_poses:
        # Extract position and euler angles
        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        euler = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

        # Modify z, pitch, yaw
        modified_position = np.array([position[0], position[1], initial_z])
        modified_euler = np.array([initial_roll, initial_pitch, euler[2]])

        # Create new pose
        new_rotation_matrix = R.from_euler('xyz', modified_euler, degrees=True).as_matrix()
        
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_rotation_matrix
        new_pose[:3, 3] = modified_position
        
        modified_processed_poses.append(new_pose)

    # De-process poses to save them in the original format
    modified_raw_poses = [deprocess_pose(p) for p in modified_processed_poses]
    
    # Save to new file
    output_file = 'data/Replica/oogway_realsense_Jan15/traj_modified.txt'
    save_poses(output_file, modified_raw_poses)
    
    print(f"Modified trajectory saved to {output_file}")

if __name__ == '__main__':
    main()
