# image_saver.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver(Node):
  """
  创建一个节点，订阅图像话题，并将指定数量的图像帧保存到文件夹中。
  """
  def __init__(self):
    super().__init__('image_saver_node')
    
    # --- 参数 ---
    self.TARGET_IMAGE_COUNT = 10
    self.OUTPUT_DIR = 'saved_images'
    # -------------
    
    self.bridge = CvBridge()
    self.image_count = 0

    # 检查并创建输出文件夹
    if not os.path.exists(self.OUTPUT_DIR):
        os.makedirs(self.OUTPUT_DIR)
        self.get_logger().info(f"Created directory: {self.OUTPUT_DIR}")

    # 创建订阅者
    self.subscription = self.create_subscription(
      Image, 
      '/camera/color/helmet_tracked', 
      self.image_callback, 
      10)
    self.subscription
    
    self.get_logger().info(f"Node started. Will save {self.TARGET_IMAGE_COUNT} images to '{self.OUTPUT_DIR}' folder.")

  def image_callback(self, msg):
    """
    回调函数，用于处理接收到的图像消息。
    """
    # 如果已经保存了足够多的图片，则不再处理新的消息
    if self.image_count >= self.TARGET_IMAGE_COUNT:
      return

    try:
      # 将ROS Image消息转换为OpenCV图像
      cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    except Exception as e:
      self.get_logger().error(f'Failed to convert image: {e}')
      return

    # 增加计数器
    self.image_count += 1
    
    # 构建文件名
    filename = os.path.join(self.OUTPUT_DIR, f'image_{self.image_count}.png')
    
    # 保存图像
    try:
      cv2.imwrite(filename, cv_image)
      self.get_logger().info(f'Saved image {self.image_count}/{self.TARGET_IMAGE_COUNT}: {filename}')
    except Exception as e:
      self.get_logger().error(f'Failed to save image: {e}')
      # 如果保存失败，将计数器减回去，以便下次重试
      self.image_count -= 1
      return

    # 检查是否已达到目标数量
    if self.image_count >= self.TARGET_IMAGE_COUNT:
      self.get_logger().info('Target image count reached. Shutting down node.')
      # 发出关闭信号，这将使 rclpy.spin() 停止阻塞
      rclpy.shutdown()

def main(args=None):
  rclpy.init(args=args)
  
  image_saver = ImageSaver()
  
  # spin() 会阻塞程序，直到 rclpy.shutdown() 被调用
  try:
    rclpy.spin(image_saver)
  except KeyboardInterrupt:
    image_saver.get_logger().info('Shutdown requested by user.')
  finally:
    # 清理资源
    image_saver.destroy_node()
    # 确保rclpy完全关闭，即使spin没有被调用
    if rclpy.ok():
        rclpy.shutdown()
    print("Node has been shut down.")

if __name__ == '__main__':
  main()