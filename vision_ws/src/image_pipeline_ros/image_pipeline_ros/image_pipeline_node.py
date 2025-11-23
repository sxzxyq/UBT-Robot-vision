# 文件路径: ~/Workspace/vision_ws/src/image_pipeline_ros/image_pipeline_ros/image_pipeline_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import torch
from hydra.core.global_hydra import GlobalHydra
import threading
import os

# --- 关键部分：从您的本地代码库导入模块 ---
# Python解释器能找到它们，是因为我们稍后会在setup.py中进行配置
# from ImagePipeline.imagepipeline.imagepipeline_module import ImagePipeline
# from ImagePipeline.imagepipeline.model_adapters import QwenVLAdapter
# from ImagePipeline.imagepipeline.utils import visualize, compute_mask_centers, visualize_centers
# from cutie.utils.get_default_model import get_default_model

class ImagePipelineNode(Node):
    def __init__(self):
        super().__init__('image_pipeline_node')
        
        # --- 路径管理 ---
        # 使用绝对路径，确保在任何地方启动节点都能找到文件
        from ImagePipeline.imagepipeline.imagepipeline_module import ImagePipeline
        from ImagePipeline.imagepipeline.model_adapters import QwenVLAdapter
        from ImagePipeline.imagepipeline.utils import visualize, compute_mask_centers, visualize_centers
        from cutie.utils.get_default_model import get_default_model
        from hydra.core.global_hydra import GlobalHydra
        
        workspace_dir = os.path.expanduser('~/Workspace/vision_ws')

        # --- ROS 2 参数声明 ---
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        # 路径现在直接指向您的src文件夹中的模型和配置文件
        self.declare_parameter('sam_checkpoint', os.path.join(workspace_dir, 'src/sam2/checkpoints/sam2.1_hiera_large.pt'))
        self.declare_parameter('sam_model_config', os.path.join(workspace_dir, 'src/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml'))
        # 建议将大的模型（如Qwen）放在一个固定的外部位置，而不是代码库里
        self.declare_parameter('qwen_model_path', '/home/nvidia/data/models/Qwen/Qwen2.5-VL-7B-Instruct') 
        self.declare_parameter('initial_instruction', '一个红色的苹果') # 初始指令
        self.declare_parameter('visualize', True) # 是否发布可视化图像
        self.declare_parameter('device', 'cuda:0') # 在Jetson上通常是 cuda:0

        # 获取参数值
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        sam_checkpoint = self.get_parameter('sam_checkpoint').get_parameter_value().string_value
        sam_model_config = self.get_parameter('sam_model_config').get_parameter_value().string_value
        qwen_model_path = self.get_parameter('qwen_model_path').get_parameter_value().string_value
        self.instruction = self.get_parameter('initial_instruction').get_parameter_value().string_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        
        self.get_logger().info(f"使用的计算设备: {self.device}")
        self.get_logger().info(f"SAM模型权重路径: {sam_checkpoint}")
        
        # --- 初始化模型与Pipeline ---
        self.bridge = CvBridge()
        self.pipeline = None
        self.model_init_lock = threading.Lock()
        # 在一个单独的线程中初始化模型，避免阻塞ROS节点启动
        self.init_thread = threading.Thread(target=self.initialize_pipeline, args=(sam_checkpoint, sam_model_config, qwen_model_path))
        self.init_thread.start()
        
        # --- ROS 2 通信接口 ---
        self.image_subscription = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.result_image_publisher = self.create_publisher(Image, '~/result_image', 10)
        self.centers_publisher = self.create_publisher(Float32MultiArray, '~/target_centers', 10)
        self.instruction_subscription = self.create_subscription(String, '~/set_instruction', self.instruction_callback, 10)
        self.reset_service = self.create_service(Trigger, '~/reset_tracking', self.reset_service_callback)
        
        self.get_logger().info("图像处理节点已启动，正在等待模型初始化...")

    def initialize_pipeline(self, sam_checkpoint, sam_model_config, qwen_model_path):
        with self.model_init_lock:
            if self.pipeline: return
            self.get_logger().info("正在初始化AI模型，请稍候...")
            try:
                with torch.no_grad():
                    if GlobalHydra().is_initialized(): GlobalHydra.instance().clear()
                    vl_adapter = QwenVLAdapter(model_path=qwen_model_path)
                    cutie_model = get_default_model()
                    self.pipeline = ImagePipeline(
                        sam_checkpoint=sam_checkpoint,
                        sam_model_config=sam_model_config,
                        cutie_model=cutie_model,
                        vl_adapter=vl_adapter
                    )
                    # 手动为您的Pipeline实例设置计算设备
                    # (您可能需要修改ImagePipeline类，使其能接收或设置device)
                    self.pipeline.device = torch.device(self.device)
                self.get_logger().info("所有模型初始化成功！")
            except Exception as e:
                self.get_logger().error(f"模型初始化失败: {e}", exc_info=True) # exc_info=True会打印详细的错误堆栈
                self.pipeline = None
    

        # ... (代码和之前一样)
        pass

    def image_callback(self, msg: Image):
        # 检查 pipeline 是否已成功初始化
        if self.pipeline is None or self.init_thread.is_alive():
            if self.init_thread.is_alive():
                 self.get_logger().warn("Models are still initializing, skipping frame.", throttle_duration_sec=5)
            else:
                 self.get_logger().error("Pipeline is not initialized, cannot process frame.", throttle_duration_sec=5)
            return

        try:
            # 将 ROS 图像消息转换为 OpenCV 图像 (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 转换为 RGB for pipeline
            rgb_image = cv_image[:, :, ::-1].copy()
            
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
            
        try:
            # 检查 pipeline 是否需要初始化跟踪
            if not self.pipeline.is_initialized:
                self.get_logger().info(f"Initializing tracking with instruction: '{self.instruction}'")
                combined_mask, bboxes = self.pipeline.initialize_with_instruction(
                    frame=rgb_image,
                    instruction=self.instruction,
                    return_bbox=True
                )
                updated_mask = combined_mask
                self.get_logger().info(f"Tracking initialized. Found {len(bboxes)} object(s).")
            else:
                # 如果已初始化，则更新跟踪
                updated_mask, _ = self.pipeline.update_masks(rgb_image)

            # 计算中心点并发布
            centers = compute_mask_centers(updated_mask, "centroid")
            if centers:
                # centers is a list of tuples [(x1, y1), (x2, y2), ...]
                flat_centers = [coord for center in centers for coord in center]
                centers_msg = Float32MultiArray()
                centers_msg.data = flat_centers
                self.centers_publisher.publish(centers_msg)

            # 可视化并发布结果图像
            if self.visualize:
                # 在原始 BGR 图像上绘制
                vis_frame = cv_image.copy()
                visualize(vis_frame, mask=updated_mask)
                visualize_centers(vis_frame, centers=centers) # 可选
                
                # 发布结果图像
                result_img_msg = self.bridge.cv2_to_imgmsg(vis_frame, "bgr8")
                result_img_msg.header = msg.header
                self.result_image_publisher.publish(result_img_msg)

        except Exception as e:
            self.get_logger().error(f"Error during pipeline processing: {e}")
            # 发生错误时重置 pipeline 状态，以便下次可以重新尝试初始化
            if self.pipeline:
                self.pipeline.reset()

    def instruction_callback(self, msg: String):
        new_instruction = msg.data
        if new_instruction != self.instruction:
            self.get_logger().info(f"Received new instruction: '{new_instruction}'. Resetting tracking.")
            self.instruction = new_instruction
            if self.pipeline:
                self.pipeline.reset() # 重置 pipeline 以使用新指令进行初始化
    
    def reset_service_callback(self, request, response):
        self.get_logger().info("Reset service called. Re-initializing tracking on next frame.")
        if self.pipeline:
            self.pipeline.reset()
        response.success = True
        response.message = "Tracking will be re-initialized on the next frame."
        return response

def main(args=None):
    rclpy.init(args=args)
    image_pipeline_node = ImagePipelineNode()
    rclpy.spin(image_pipeline_node)
    image_pipeline_node.destroy_node()
    rclpy.shutdown()
    
class ColorInverterNodeTest(Node):
    """
    一个完全独立的测试节点，用于验证ROS 2包的基本功能。
    它只做一件事：订阅图像，翻转颜色，然后发布。
    """
    def __init__(self):
        super().__init__('color_inverter_test_node')
        
        # 使用硬编码的话题名，因为这只是一个简单的内部测试
        input_topic = '/camera/color/image_raw'
        output_topic = '~/inverted_image_test'
        
        self.get_logger().info(f"测试节点启动，订阅: '{input_topic}'")
        self.get_logger().info(f"测试节点启动，发布到: '{output_topic}'")

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, input_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(Image, output_topic, 10)

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            inverted_image = 255 - cv_image
            output_msg = self.bridge.cv2_to_imgmsg(inverted_image, "bgr8")
            output_msg.header = msg.header
            self.publisher.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f"测试节点处理图像时出错: {e}")

def main_test(args=None):
    """
    专门用于启动测试节点的main函数。
    """
    rclpy.init(args=args)
    test_node = ColorInverterNodeTest()
    print("--- 颜色翻转测试节点正在运行。按 Ctrl+C 退出。 ---")
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        print("--- 测试节点关闭。 ---")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()    

if __name__ == '__main__':
    #main()
    main_test()