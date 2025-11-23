# 文件: helmet_tf_transformer.py
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import tf2_sensor_msgs.tf2_sensor_msgs # 用于点云转换
from tf2_geometry_msgs import do_transform_point # 用于PointStamped转换

class HelmetTransformerNode(Node):
    """
    订阅头盔的点云和位置，并将它们转换到目标坐标系（如 'pelvis'）。
    """
    def __init__(self):
        super().__init__('helmet_transformer_node')
        self.get_logger().info('Helmet TF Transformer Node Started.')

        # ======================= 参数定义 =======================
        # 定义我们想要转换到的目标坐标系
        self.target_frame = 'pelvis' 
        # =======================================================

        # TF2 监听器，用于获取坐标变换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 订阅原始的点云和位置话题
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/helmet/points',
            self.pointcloud_callback,
            10
        )
        self.position_sub = self.create_subscription(
            PointStamped,
            '/helmet/position',
            self.position_callback,
            10
        )
        self.filtered_position_sub = self.create_subscription( # <<< 新增
            PointStamped,
            '/helmet/position_filtered', # 订阅滤波后的话题
            self.filtered_position_callback, # 使用一个新的回调函数
            10
        )

        # 创建发布器，用于发布转换后的话题
        self.transformed_pointcloud_pub = self.create_publisher(
            PointCloud2, 
            '/helmet/points_transformed', 
            10
        )
        self.transformed_position_pub = self.create_publisher(
            PointStamped,
            '/helmet/position_transformed',
            10
        )
        self.transformed_filtered_position_pub = self.create_publisher( # <<< 新增
            PointStamped,
            '/helmet/position_filtered_transformed', # 发布到新的话题，方便对比
            10
        )
        

    def pointcloud_callback(self, msg: PointCloud2):
        """处理接收到的点云消息。"""
        if not msg.data:
            self.get_logger().debug("Received an empty point cloud, skipping transformation.")
            return
        
        source_frame = msg.header.frame_id
        
        # 如果源坐标系和目标坐标系相同，则无需转换
        if source_frame == self.target_frame:
            self.transformed_pointcloud_pub.publish(msg)
            return

        try:
            # 查找从源坐标系到目标坐标系的变换
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time(), # 获取最新的可用变换
                timeout=Duration(seconds=1.0)
            )
            
            # 使用tf2_sensor_msgs库提供的函数高效地转换整个点云
            cloud_transformed = tf2_sensor_msgs.do_transform_cloud(msg, transform)
            
            # 发布转换后的点云
            self.transformed_pointcloud_pub.publish(cloud_transformed)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"Failed to transform PointCloud from '{source_frame}' to '{self.target_frame}': {e}",
                throttle_duration_sec=2.0
            )

    def position_callback(self, msg: PointStamped):
        """处理接收到的位置消息。"""
        source_frame = msg.header.frame_id

        if source_frame == self.target_frame:
            self.transformed_position_pub.publish(msg)
            return
        
        try:
            # 查找变换
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )

            # PointStamped 的转换非常直接
            point_transformed = do_transform_point(msg, transform)

            # 发布转换后的位置
            self.transformed_position_pub.publish(point_transformed)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"Failed to transform PointStamped from '{source_frame}' to '{self.target_frame}': {e}",
                throttle_duration_sec=2.0
            )

    def filtered_position_callback(self, msg: PointStamped): # <<< 新增函数
        """
        处理接收到的【滤波后】的位置消息。
        逻辑与 `position_callback` 完全相同，只是发布到不同的话题。
        """
        source_frame = msg.header.frame_id
        if source_frame == self.target_frame:
            self.transformed_filtered_position_pub.publish(msg)
            return
        
        try:
            # 查找变换
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )

            # 应用变换
            point_transformed = do_transform_point(msg, transform)

            # 发布到【滤波后+变换后】的话题
            self.transformed_filtered_position_pub.publish(point_transformed)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"Failed to transform FILTERED PointStamped from '{source_frame}' to '{self.target_frame}': {e}",
                throttle_duration_sec=2.0
            )
            
def main(args=None):
    rclpy.init(args=args)
    transformer_node = HelmetTransformerNode()
    try:
        rclpy.spin(transformer_node)
    except KeyboardInterrupt:
        pass
    finally:
        transformer_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()