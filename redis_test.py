import redis
import pickle
import numpy as np
import datetime




from scipy.spatial.transform import Rotation # to process quaternion from vicon bridge 




with open('/home/iconlab/Documents/codes/ros2_turtlebot_icon/turtlebot_icon_ws/saved_data/oogway_single/pose_47.pkl', 'rb') as f:
        vicon_data = pickle.load(f)

#vicon_data = pickle.loads(loaded_data)  # vicon_data is now a PoseStamped object
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
print(transformation_matrix)


# # Redis connection
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)




# while True:
#     # send data
#     # send data
#     theta_i =  np.random.rand(1633360).astype(np.float32)
#     uncertainty_i  =  np.random.rand(1633360).astype(np.float32)

#     msg = {'theta_i':theta_i, 'uncertainty_i':uncertainty_i}
#     pickled_data = pickle.dumps(msg)
#     redis_client.set('agent_i', pickled_data )


#     # get data
#     agent_j = redis_client.get('agent_j')
#     if agent_j:
#         print(f"{datetime.datetime.now()}: receive agent j!")
#         agent_j = pickle.loads(agent_j)
#         array_1 = agent_j['theta_j']
#         array_2 = agent_j['uncertainty_j']

#         print(f'theta_j = {array_1.shape}, uncertainty_j= {array_2.shape}')