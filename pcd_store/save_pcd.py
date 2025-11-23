#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np

class PCDSaver(Node):
    """
    一个简单的ROS 2节点，用于订阅一个PointCloud2话题，
    并将接收到的第一条消息保存为.pcd文件，然后自动退出。
    """
    def __init__(self):
        super().__init__('pcd_saver_node')
        
        # 定义要订阅的话题名称
        topic_name = '/helmet/points'
        #topic_name = '/camera/depth/points'
        
        self.subscription = self.create_subscription(
            PointCloud2,
            topic_name,
            self.listener_callback,
            10) # QoS depth
            
        self.get_logger().info(f"PCD Saver node started. Waiting for a message on '{topic_name}'...")
        self.saved = False # 一个标志，确保我们只保存一次

    def listener_callback(self, msg: PointCloud2):
        # 如果已经保存过了，就直接返回，不再处理后续消息
        if self.saved:
            return

        # 检查点云是否为空
        if msg.height * msg.width == 0:
            self.get_logger().warn("Received an empty point cloud. Waiting for a valid one.")
            return

        self.get_logger().info(f"Received a PointCloud2 message with {msg.width * msg.height} points.")
        
        try:
            # 使用 sensor_msgs_py.point_cloud2 将消息转换为Numpy数组
            # 我们只关心 x, y, z 字段
            points_numpy = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
            
            # --- 手动构建 .pcd 文件内容 ---
            
            # 1. 构建PCD文件头
            num_points = len(points_numpy)
            header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""
            
            # 2. 将点数据格式化为字符串
            # 使用列表推导式和字符串格式化，高效地构建文件主体
            body_lines = [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points_numpy]
            body = "\n".join(body_lines)
            
            # 3. 写入文件
            filename = "helmet_cloud.pcd"
            with open(filename, 'w') as f:
                f.write(header + body)
            
            self.get_logger().info(f"SUCCESS! Point cloud data saved to '{filename}'.")
            
            # 4. 设置标志并关闭节点
            self.saved = True
            self.get_logger().info("Shutting down node...")
            # 调用 rclpy.shutdown() 会让 spin() 停止阻塞
            self.context.shutdown()

        except Exception as e:
            self.get_logger().error(f"Failed to save PCD file: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            # 即使出错也关闭节点，避免持续报错
            self.context.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    pcd_saver = PCDSaver()
    
    try:
        # rclpy.spin() 会阻塞在这里，直到节点被关闭
        rclpy.spin(pcd_saver)
    except KeyboardInterrupt:
        pcd_saver.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        # 清理资源
        pcd_saver.destroy_node()
        # 确保rclpy完全关闭
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()