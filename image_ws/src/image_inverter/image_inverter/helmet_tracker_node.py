# 文件: ~/Workspace/image_ws/src/image_inverter/image_inverter/helmet_tracker_node.py
import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool 
import numpy as np 

# 导入推迟到 main 函数中

class HelmetTrackerNode(Node):
    """
    一个ROS 2节点，使用 HelmetTracker 类来实现检测、分割和跟踪。
    """
    def __init__(self):
        super().__init__('helmet_tracker_node')
        self.get_logger().info('Node created. Waiting for tracker setup...')
        self.tracker = None
        self.bridge = None
        self.subscription = None
        self.publisher = None
        self.trigger_subscription = None

    def setup(self, tracker_instance):
        self.get_logger().info('Setting up ROS communications...')
        self.tracker = tracker_instance
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )
        self.publisher = self.create_publisher(
            Image, '/camera/color/helmet_tracked', 10 # 新的话题名
        )
        self.mask_publisher = self.create_publisher(Image, '/camera/mask/helmet', 10)
        self.trigger_subscription = self.create_subscription(
            Bool, 
            '/trigger_reset', # 这是我们的控制话题
            self.trigger_callback, 
            10
        )
        self.get_logger().info('Helmet Tracker Node has been fully started.')
        
    def trigger_callback(self, msg):
        """
        当接收到 /trigger_reset 话题的消息时，调用此函数。
        """
        if msg.data: # 如果消息内容为 True
            self.get_logger().warn('>>> Reset trigger received! <<<')
            if self.tracker:
                self.tracker.reset() # 调用核心逻辑类中的 reset 方法
                self.get_logger().info('Tracker state has been reset to SEARCHING.')
            else:
                self.get_logger().error('Tracker is not initialized, cannot reset.')

    def image_callback(self, msg):
        if not self.tracker:
            return

        self.get_logger().info('Receiving image frame...')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # --- 调用核心处理逻辑 ---
        try:
            # 将帧交给 HelmetTracker 处理，它会根据内部状态决定做什么
            processed_frame, mask_frame = self.tracker.process_frame(cv_image)
        except Exception as e:
            self.get_logger().error(f"An error occurred in tracker.process_frame: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            # 发生严重错误时，重置跟踪器
            self.tracker.reset()
            processed_frame = cv_image # 发布原图
            mask_frame = np.zeros(cv_image.shape[:2], dtype=np.uint8)

        # --- 发布处理后的图像 ---
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, 'bgr8')
            processed_msg.header.stamp = self.get_clock().now().to_msg()
            processed_msg.header.frame_id = msg.header.frame_id
            self.publisher.publish(processed_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert processed image for publishing: {e}')
            
        try:
            # 掩码是单通道8位图像
            mask_msg = self.bridge.cv2_to_imgmsg(mask_frame, 'mono8')
            mask_msg.header.stamp = self.get_clock().now().to_msg()
            mask_msg.header.frame_id = msg.header.frame_id
            self.mask_publisher.publish(mask_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert mask image for publishing: {e}')


def main(args=None):
    # ========================== 关键逻辑区 - 开始 ==========================
    # import os
    # 需要添加根目录，以便能找到 sam2, Cutie, ImagePipeline
    image_pipeline_base_path = os.path.abspath(os.path.expanduser('~/Workspace/imagepipeline_conda'))
    if image_pipeline_base_path not in sys.path:
        sys.path.insert(0, image_pipeline_base_path)
        print(f"INFO: Added '{image_pipeline_base_path}' to Python sys.path.")
    
    # 导入我们自己的模块
    try:
        # 从新文件中导入新类
        from image_inverter.helmet_tracker import HelmetTracker
        print("INFO: Successfully imported HelmetTracker.")
    except ImportError as e:
        print("="*80, "\nFATAL ERROR: Failed to import HelmetTracker.", f"\nError details: {e}\n", "="*80)
        return
    # ========================== 关键逻辑区 - 结束 ==========================

    rclpy.init(args=args)
    
    tracker_node = HelmetTrackerNode()
    
    try:
        tracker_node.get_logger().info('Loading all AI models... (this may take a long time)')
        
        # --- 配置模型路径 ---
        # 确保这些路径对于你的环境是正确的
        sam_checkpoint_path = os.path.join(image_pipeline_base_path, "sam2/checkpoints/sam2.1_hiera_large.pt")
        #sam_model_config_path = os.path.join(image_pipeline_base_path, "sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
        sam_model_config_logic_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        tracker_instance = HelmetTracker(
            sam_checkpoint=sam_checkpoint_path,
            sam_model_config=sam_model_config_logic_path
        )
        tracker_node.get_logger().info('All AI models loaded successfully.')
    except Exception as e:
        tracker_node.get_logger().fatal(f"Failed to create HelmetTracker instance. Aborting. Error: {e}")
        import traceback
        tracker_node.get_logger().error(traceback.format_exc())
        tracker_node.destroy_node()
        rclpy.shutdown()
        return

    tracker_node.setup(tracker_instance)
    
    try:
        rclpy.spin(tracker_node)
    except KeyboardInterrupt:
        pass
    finally:
        tracker_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()