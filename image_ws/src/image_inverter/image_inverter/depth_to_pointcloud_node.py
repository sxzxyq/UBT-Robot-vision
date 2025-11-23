# 文件: ~/Workspace/image_ws/src/image_inverter/image_inverter/depth_to_pointcloud_node.py (修正版)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import message_filters
import numpy as np

class DepthToPointCloudNode(Node):
    """
    一个ROS 2节点，通过同步订阅深度图像和相机信息，将其转换为点云数据发布。
    """
    def __init__(self):
        super().__init__('depth_to_pointcloud_node')

        self.get_logger().info('Initializing Depth to PointCloud Node with CameraInfo subscription...')
        
        # ======================= 关键修正 - 开始 =======================
        # 1. 在构造函数 __init__ 中声明一次参数
        #    这样它只会在节点启动时执行一次
        self.declare_parameter('camera.depth_scale', 1000.0)
        # ======================= 关键修正 - 结束 =======================
        
        self.bridge = CvBridge()
        self.camera_info_received = False
        
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/depth/camera_info')
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.info_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        
        self.publisher = self.create_publisher(PointCloud2, '/camera/depth/points', 10)
        self.get_logger().info('Node started. Waiting for synchronized depth and camera_info messages...')

    def synchronized_callback(self, depth_msg, info_msg):
        """
        当接收到时间同步的深度图和相机信息时，此回调函数被调用。
        """
        if not self.camera_info_received:
            self.get_logger().info('First synchronized messages received.')
            self.camera_info_received = True

        try:
            self.fx = info_msg.k[0]
            self.fy = info_msg.k[4]
            self.cx = info_msg.k[2]
            self.cy = info_msg.k[5]
            self.width = info_msg.width
            self.height = info_msg.height
            
            # ======================= 关键修正 - 开始 =======================
            # 2. 在回调函数中，只获取参数的值
            #    这样每次回调都会读取最新的参数值（如果它在运行时被改变了）
            depth_scale = self.get_parameter('camera.depth_scale').get_parameter_value().double_value
            # ======================= 关键修正 - 结束 =======================

            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')

            points = []
            for v in range(self.height):
                for u in range(self.width):
                    depth = depth_image[v, u]
                    if depth == 0:
                        continue
                    
                    z = float(depth) / depth_scale
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    points.append([float(x), float(y), float(z)])

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            
            header = depth_msg.header
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points)
            
            self.publisher.publish(point_cloud_msg)
            self.get_logger().info(f'Published point cloud with {len(points)} points.', throttle_duration_sec=1.0)

        except Exception as e:
            self.get_logger().error(f'Error in synchronized_callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    depth_to_pointcloud_node = DepthToPointCloudNode()
    rclpy.spin(depth_to_pointcloud_node)
    depth_to_pointcloud_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()