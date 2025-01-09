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

class sub_redis_node(Node):
    def __init__(self, bot_name):
        super().__init__('sub_redis_node')

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.maxDepth = 10*1000

        # Redis connection
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

        self.subscription1 = self.create_subscription(
            Image, f'/{bot_name}/oakd/rgb/image_raw', self.rgb_image_callback, qos_profile_sensor_data)
        self.subscription2 = self.create_subscription(
            Image, f'/{bot_name}/oakd/stereo/image_raw', self.depth_image_callback, qos_profile_sensor_data)
        self.subscription3 = self.create_subscription(
            PoseStamped, f'/vicon/{bot_name}/{bot_name}/pose', self.vicon_data_callback, qos_profile_sensor_data)


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

    image_blender = sub_redis_node(bot_name)
    rclpy.spin(image_blender)
    image_blender.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()