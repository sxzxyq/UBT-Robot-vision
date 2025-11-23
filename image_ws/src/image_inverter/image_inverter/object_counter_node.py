#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import re # 导入正则表达式库，用于从文本中提取数字

# 导入基础检测器类 (复用你的VLM逻辑)
try:
    from .helmet_detector_base import HelmetDetectorBase
except ImportError:
    import sys
    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_path)
    from helmet_detector_base import HelmetDetectorBase

# 导入新的服务类型
from image_inverter_interfaces.srv import CountObjects

class ObjectCounterNode(Node):
    """
    一个提供通用物体计数服务的节点。
    1. 持续接收最新的图像。
    2. 当接收到服务请求时，使用请求中的prompt对最新图像进行VLM查询。
    3. 从VLM的文本响应中解析出数字，并作为服务响应返回。
    """
    def __init__(self):
        super().__init__('object_counter_node')

        # 参数初始化
        self.declare_parameter('api_key', '')
        self.api_key = self.get_parameter('api_key').get_parameter_value().string_value
        
        self.get_logger().info("--- Object Counter Service Node Initializing ---")
        
        # 状态与数据存储
        self.latest_frame = None
        self.bridge = CvBridge()
        self.processing_in_progress = False

        # 初始化VLM检测器 (复用现有模块)
        try:
            # 注意: 虽然类名叫HelmetDetector，但我们实际上是在使用它通用的VLM能力
            self.vlm_handler = HelmetDetectorBase(api_key=self.api_key if self.api_key else None)
            self.get_logger().info("✅ VLM Handler (HelmetDetectorBase) initialized successfully.")
        except Exception as e:
            self.get_logger().fatal(f"❌ Failed to initialize VLM Handler: {e}")
            rclpy.shutdown()
            return

        # 创建服务服务端
        self.srv = self.create_service(
            CountObjects,
            '/count_objects', # 这是新的服务名称
            self.count_objects_callback) # 这是处理请求的回调函数
        
        # 订阅图像流 (你可以根据需要修改话题名称)
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw', # 订阅原始图像，更通用
            self.image_callback,
            10)
            
        self.get_logger().info("--- Object Counter Service is ready to receive requests ---")

    def image_callback(self, msg: Image):
        """存储最新的图像帧。"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            
    def count_objects_callback(self, request, response):
        """
        当收到计数服务请求时，执行此函数。
        """
        if self.processing_in_progress:
            self.get_logger().warn("Processing already in progress. Rejecting new request.")
            response.success = False
            response.count = -1
            response.message = "Server is busy."
            return response

        self.processing_in_progress = True
        # 从请求中获取用户指定的prompt
        prompt = request.prompt
        self.get_logger().info(f"🤖 Received counting request with prompt: '{prompt}'")

        if self.latest_frame is None:
            self.get_logger().error("Counting requested, but no image frame is available.")
            response.success = False
            response.count = -1
            response.message = "No image available for counting."
            self.processing_in_progress = False
            return response

        # --- 核心计数逻辑 ---
        frame_to_process = self.latest_frame.copy()

        try:
            base64_image = self.vlm_handler._image_to_base64(frame_to_process)
            input_data = self.vlm_handler.vl_adapter.prepare_input(text=prompt, image_url=f"data:image/jpeg;base64,{base64_image}")
            # 增加max_tokens以确保能得到完整的句子
            vlm_response_text, _ = self.vlm_handler.vl_adapter.generate_response(input_data, max_tokens=50)
            
            self.get_logger().info(f"VLM raw response: '{vlm_response_text.strip()}'")

            # --- 从VLM响应中解析数字 ---
            numbers_found = re.findall(r'\d+', vlm_response_text)
            
            if numbers_found:
                # 假设第一个找到的数字就是我们想要的数量
                count = int(numbers_found[0])
                response.success = True
                response.count = count
                response.message = f"Successfully parsed count. VLM response: '{vlm_response_text.strip()}'"
                self.get_logger().info(f"✅ Parsed count: {count}")
            else:
                # 如果VLM的回答中没有数字
                response.success = False
                response.count = -1
                response.message = f"Failed to parse a number from VLM response: '{vlm_response_text.strip()}'"
                self.get_logger().warn("⚠️ Could not find any number in the VLM response.")

        except Exception as e:
            self.get_logger().error(f"An error occurred during VLM query: {e}")
            response.success = False
            response.count = -1
            response.message = f"Error during VLM query: {e}"
        finally:
            self.processing_in_progress = False
            return response

def main(args=None):
    rclpy.init(args=args)
    node = ObjectCounterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()