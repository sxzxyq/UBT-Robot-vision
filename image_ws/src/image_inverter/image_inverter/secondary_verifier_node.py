#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

# 导入基础检测器类 (这部分不变)
try:
    from .helmet_detector_base import HelmetDetectorBase
except ImportError:
    import sys
    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_path)
    from helmet_detector_base import HelmetDetectorBase

# ======================= 修改点 1: 导入新的服务类型 =======================
from image_inverter_interfaces.srv import TriggerVerification

class VerifierServiceNode(Node): # 建议改个新名字，更清晰
    """
    一个提供安全帽验证服务的节点。
    1. 持续接收最新的带掩码图像。
    2. 当接收到服务请求时，对最新图像进行VLM验证。
    3. 将验证结果作为服务的响应返回。
    """
    def __init__(self):
        super().__init__('verifier_service_node')

        # 参数初始化 (不变)
        self.declare_parameter('api_key', '')
        self.api_key = self.get_parameter('api_key').get_parameter_value().string_value
        
        self.get_logger().info("--- Verifier Service Node Initializing ---")
        
        # 状态与数据存储 (不变)
        self.latest_frame = None
        self.bridge = CvBridge()
        self.verification_in_progress = False

        # 初始化VLM检测器 (不变)
        try:
            self.helmet_detector = HelmetDetectorBase(api_key=self.api_key if self.api_key else None)
            self.get_logger().info("✅ HelmetDetectorBase initialized successfully.")
        except Exception as e:
            self.get_logger().fatal(f"❌ Failed to initialize HelmetDetectorBase: {e}")
            rclpy.shutdown()
            return

        # ======================= 修改点 2: 改造ROS通信接口 =======================
        # 移除了 trigger_subscriber 和 status_publisher
        # 增加了一个 Service Server

        # 1. 创建服务服务端
        self.srv = self.create_service(
            TriggerVerification,
            '/trigger_verification', # 这是服务名称
            self.verification_callback) # 这是处理请求的回调函数
        
        # 2. 订阅图像流 (不变)
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/color/helmet_tracked',
            self.image_callback,
            10)
            
        self.get_logger().info("--- Verifier Service is ready to receive requests ---")

    def image_callback(self, msg: Image):
        """(不变) 存储最新的图像帧。"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            
    # ======================= 修改点 3: 实现服务回调函数 =======================
    # 这个函数取代了旧的 trigger_callback 和 run_verification
    def verification_callback(self, request, response):
        """
        当收到服务请求时，执行此函数。
        :param request: 服务请求对象 (在我们的例子中为空)
        :param response: 服务响应对象 (我们需要填充并返回它)
        """
        if self.verification_in_progress:
            self.get_logger().warn("Verification already in progress. Rejecting new request.")
            response.is_compliant = False # 可以返回一个默认的失败状态
            response.message = "Server is busy."
            return response

        self.verification_in_progress = True
        self.get_logger().info("🤖 Received a verification request. Starting process...")

        if self.latest_frame is None:
            self.get_logger().error("Verification requested, but no image frame is available.")
            response.is_compliant = False
            response.message = "No image available for verification."
            self.verification_in_progress = False
            return response

        if self.helmet_detector.vl_adapter is None:
            self.get_logger().error("VLM Adapter is not available.")
            response.is_compliant = False
            response.message = "VLM is not initialized."
            self.verification_in_progress = False
            return response

        # --- 核心验证逻辑 (从旧的 run_verification 移入) ---
        frame_to_verify = self.latest_frame.copy()
        prompt = (
            "You are a safety compliance verifier. In the provided image, a person has been highlighted with a visual mask. "
            "Your task is to determine if THIS SPECIFIC PERSON is wearing a safety helmet. "
            "Answer ONLY with the word 'YES' if they are wearing a helmet, or 'NO' if they are not."
        )

        try:
            base64_image = self.helmet_detector._image_to_base64(frame_to_verify)
            input_data = self.helmet_detector.vl_adapter.prepare_input(text=prompt, image_url=f"data:image/jpeg;base64,{base64_image}")
            vlm_response, _ = self.helmet_detector.vl_adapter.generate_response(input_data, max_tokens=10)
            
            self.get_logger().info(f"VLM raw response: '{vlm_response.strip()}'")

            # --- 填充响应对象 ---
            if "YES" in vlm_response.upper():
                response.is_compliant = True
                response.message = "Verification successful: Helmet detected."
                self.get_logger().info("✅ Verification Result: Helmet DETECTED.")
            else:
                response.is_compliant = False
                response.message = "Verification failed: Helmet not detected."
                self.get_logger().info("❌ Verification Result: Helmet NOT detected.")
        except Exception as e:
            self.get_logger().error(f"An error occurred during VLM verification: {e}")
            response.is_compliant = False
            response.message = f"Error during verification: {e}"
        finally:
            self.verification_in_progress = False
            # --- 关键: 返回填充好的响应对象 ---
            return response

def main(args=None):
    rclpy.init(args=args)
    node = VerifierServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()