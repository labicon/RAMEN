import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import sys
import redis
import pickle  # For serializing the image data
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import datetime

class sub_redis_node(Node):
    def __init__(self, bot_name, other_bot_name):
        super().__init__('sub_redis_node')
        # define some variable
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.maxDepth = 10*1000
        consensus_publish_waitTime = 20. # don't send it fastetr than it can be processed

        # Redis connection
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

        # images + VICON sub
        self.subscription1 = self.create_subscription(
            Image, f'/{bot_name}/oakd/rgb/image_raw', self.rgb_image_callback, qos_profile_sensor_data)
        self.subscription2 = self.create_subscription(
            Image, f'/{bot_name}/oakd/stereo/image_raw', self.depth_image_callback, qos_profile_sensor_data)
        self.subscription3 = self.create_subscription(
            PoseStamped, f'/vicon/{bot_name}/{bot_name}/pose', self.vicon_data_callback, qos_profile_sensor_data)

        # for consensus
        self.subscription4 = self.create_subscription(
            Float32MultiArray, f'/{other_bot_name}/consensus', self.consensus_callback, qos_profile_sensor_data)          
        self.publisher = self.create_publisher(Float32MultiArray, f'/{bot_name}/consensus', qos_profile_sensor_data)
        self.timer = self.create_timer(consensus_publish_waitTime, self.publish_data)  # calls the publish_data functionck every x seconds


    def rgb_image_callback(self, msg: Image):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") # get uint8
        self.send_data_to_redis('rgb_image', cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
        self.blend_and_show()


    def depth_image_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1") # get unint16
        self.depth_image = np.clip(self.depth_image, 0, self.maxDepth)
        self.send_data_to_redis('depth_image', self.depth_image)
        self.blend_and_show()
    

    def vicon_data_callback(self, msg: PoseStamped):
        self.vicon_data = msg  # Store the latest ViconData message
        self.send_data_to_redis('vicon_data', self.vicon_data)
        self.blend_and_show()

    
    def consensus_callback(self, msg: Float32MultiArray):
        print(f"{datetime.datetime.now()}:    consensus receive")
        data = np.array(msg.data).reshape((msg.layout.dim[0].size, msg.layout.dim[1].size))
        theta_j = data[0, :]
        uncertainty_j = data[1, :]
        agent_j = {'theta_j':theta_j, 'uncertainty_j':uncertainty_j}
        agent_j_pickled = pickle.dumps(agent_j)
        self.redis_client.set('agent_j', agent_j_pickled)



    def publish_data(self):
        # publish
        agent_i =  self.redis_client.get('agent_i')
        if agent_i:
            print(f"{datetime.datetime.now()}:    consensus send")
            agent_i_pickled = pickle.loads(agent_i)
            theta_i = agent_i_pickled['theta_i']
            uncertainty_i = agent_i_pickled['uncertainty_i']

            # Combine the arrays into a 2D array
            combined_data = np.vstack((theta_i, uncertainty_i))  # Stack vertically

            # Create Float32MultiArray message
            msg = Float32MultiArray()
            msg.data = combined_data.flatten().tolist()  # Flatten and convert to list

            # Set the dimensions (assuming theta_i and uncertainty_i have the same length)
            """
                Given 2D array:
                    1 2 3
                    4 5 6
                    7 8 9
                Flatten it into 1D: 1 2 3 4 5 6 7 8 9
                dim[0] row: size = 3, stride = 3 (to get to next row, need to step over 3 elements in 1D array)
                dim[1] col: size = 3, stride = 1
            """
            msg.layout.dim = [
                MultiArrayDimension(label="row", size=2, stride=len(theta_i)),
                MultiArrayDimension(label="column", size=len(theta_i), stride=1)
            ]
            # Publish the message
            self.publisher.publish(msg)
    
    


    def send_data_to_redis(self, key, data):
        # Serialize the message using pickle
        pickled_data = pickle.dumps(data)
        # Store in Redis
        self.redis_client.set(key, pickled_data)


    def blend_and_show(self):
        if self.rgb_image is not None and self.depth_image is not None:
            depth_image_8bit = (self.depth_image * 255. / self.maxDepth).astype(np.uint8)
            depth_3ch = cv2.cvtColor(depth_image_8bit, cv2.COLOR_GRAY2BGR)
            depth_3ch = cv2.applyColorMap(depth_3ch, cv2.COLORMAP_JET)
            blended_image = cv2.addWeighted(self.rgb_image, 0.7, depth_3ch, 0.3, 0)
            cv2.imshow('Blended Image', blended_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    bot_name = 'miriel'  # Default bot name
    if len(sys.argv) > 1:
        bot_name = sys.argv[1]
        ohter_bot_name = sys.argv[2]

    image_blender = sub_redis_node(bot_name, ohter_bot_name)
    rclpy.spin(image_blender)
    image_blender.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()