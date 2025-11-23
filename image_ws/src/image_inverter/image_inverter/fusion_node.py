# 文件: ~/Workspace/image_ws/src/image_inverter/image_inverter/fusion_node.py (最终修正版)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import PointStamped
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import message_filters
import numpy as np

class FusionNode(Node):
    """
    融合分割掩码和点云，计算并发布目标物体的位置。
    """
    def __init__(self):
        super().__init__('fusion_node')
        self.get_logger().info('Initializing Fusion Node...')

        self.bridge = CvBridge()
        self.camera_info = None

        # self.info_sub = self.create_subscription(
        #     CameraInfo,
        #     '/camera/depth/camera_info',
        #     self.camera_info_callback,
        #     rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        # )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.camera_info_callback,
            10  # 使用一个简单的队列大小即可，它会自动使用兼容的默认QoS
        )

        self.mask_sub = message_filters.Subscriber(self, Image, '/camera/mask/helmet')
        self.points_sub = message_filters.Subscriber(self, PointCloud2, '/camera/depth/points')
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.mask_sub, self.points_sub], 
            queue_size=10, 
            slop=0.2
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        
        self.helmet_points_pub = self.create_publisher(PointCloud2, '/helmet/points', 10)
        self.helmet_position_pub = self.create_publisher(PointStamped, '/helmet/position', 10)

        self.get_logger().info('Node started. Waiting for camera info and synchronized messages...')

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(f"Received camera intrinsics (fx: {msg.k[0]}, fy: {msg.k[4]})")
            self.destroy_subscription(self.info_sub)

    def synchronized_callback(self, mask_msg, points_msg):
        if self.camera_info is None:
            self.get_logger().warn("Waiting for camera info, skipping fusion...")
            return

        try:
            mask = self.bridge.imgmsg_to_cv2(mask_msg, 'mono8')
            points_structured_array = point_cloud2.read_points_numpy(
                points_msg, 
                field_names=("x", "y", "z")
            )
            #             # ======================= 诊断代码 - 开始 =======================
            # self.get_logger().info("\n" + "="*50 + "\n" +
            #                        "            POINT CLOUD DIAGNOSTICS            " + "\n" +
            #                        "="*50)
            
            # # 1. 打印变量的类型
            # self.get_logger().info(f"Type of points_structured_array: {type(points_structured_array)}")
            
            # # 2. 如果它是Numpy数组，打印它的形状(shape)和数据类型(dtype)
            # if isinstance(points_structured_array, np.ndarray):
            #     self.get_logger().info(f"  - Shape: {points_structured_array.shape}")
            #     self.get_logger().info(f"  - DType: {points_structured_array.dtype}")
                
            # # 3. 打印数组的前几个元素，看看它的内容长什么样
            # self.get_logger().info(f"First 3 elements:\n{points_structured_array[:3]}")
            
            # self.get_logger().info("="*50 + "\n")
            # 如果没有点，则直接返回
            if points_structured_array.size == 0:
                return

            # 将结构化数组转换为一个标准的 (N, 3) 浮点数数组
            # 我们分别提取 x, y, z 列，然后用 np.stack 将它们堆叠起来
            # points_3d = np.stack(
            #     [points_structured_array['x'], points_structured_array['y'], points_structured_array['z']], 
            #     axis=-1
            # )
            points_3d = point_cloud2.read_points_numpy(
                points_msg, 
                field_names=("x", "y", "z")
            )
            # points_generator = point_cloud2.read_points(points_msg, field_names=("x", "y", "z"), skip_nans=True)
            # points_3d = np.array(list(points_generator))

            if len(points_3d) == 0:
                return

            fx, fy, cx, cy = self.camera_info.k[0], self.camera_info.k[4], self.camera_info.k[2], self.camera_info.k[5]
            
            z_values = points_3d[:, 2]
            valid_z_indices = z_values > 1e-6
            points_3d_valid = points_3d[valid_z_indices]
            z_values_valid = z_values[valid_z_indices]

            u_coords = (fx * points_3d_valid[:, 0] / z_values_valid + cx).astype(int)
            v_coords = (fy * points_3d_valid[:, 1] / z_values_valid + cy).astype(int)
            
            h, w = mask.shape
            in_bounds_indices = (u_coords >= 0) & (u_coords < w) & (v_coords >= 0) & (v_coords < h)
            
            u_in_bounds = u_coords[in_bounds_indices]
            v_in_bounds = v_coords[in_bounds_indices]
            points_3d_in_bounds = points_3d_valid[in_bounds_indices]

            mask_values = mask[v_in_bounds, u_in_bounds]
            helmet_indices = mask_values > 0
            helmet_points = points_3d_in_bounds[helmet_indices]

            if len(helmet_points) < 10:
                self.get_logger().info("Not enough points in mask.", throttle_duration_sec=2.0)
                # 即使没有点，也发布一个空点云，表示当前没有检测到目标
                empty_header = points_msg.header
                empty_cloud_msg = point_cloud2.create_cloud(empty_header, [], [])
                self.helmet_points_pub.publish(empty_cloud_msg)
                return

            header = points_msg.header
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            helmet_cloud_msg = point_cloud2.create_cloud(header, fields, helmet_points.tolist())
            self.helmet_points_pub.publish(helmet_cloud_msg)
            
            center_point = np.mean(helmet_points, axis=0)
            point_stamped_msg = PointStamped(header=header)
            point_stamped_msg.point.x, point_stamped_msg.point.y, point_stamped_msg.point.z = float(center_point[0]), float(center_point[1]), float(center_point[2])
            self.helmet_position_pub.publish(point_stamped_msg)

            self.get_logger().info(f"Published helmet position at [{center_point[0]:.3f}, {center_point[1]:.3f}, {center_point[2]:.3f}] with {len(helmet_points)} points.", throttle_duration_sec=1.0)
            
        except Exception as e:
            self.get_logger().error(f'Error in fusion callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

# ======================= 在这里添加缺失的 main 函数 =======================
def main(args=None):
    """
    主函数：程序的入口点。
    """
    rclpy.init(args=args)
    
    fusion_node = FusionNode()
    
    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 销毁节点并关闭rclpy
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
# ====================================================================