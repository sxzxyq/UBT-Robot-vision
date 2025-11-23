#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import time

class ImageSaver:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('image_saver_node', anonymous=True)
        
        # 创建一个CvBridge实例
        self.bridge = CvBridge()
        
        # 定义保存路径
        self.output_dir_raw = os.path.join(os.path.expanduser('~'), 'ros_images', 'image_raw')
        self.output_dir_tracked = os.path.join(os.path.expanduser('~'), 'ros_images', 'helmet_tracked')
        
        # 如果目录不存在，则创建它
        if not os.path.exists(self.output_dir_raw):
            os.makedirs(self.output_dir_raw)
        if not os.path.exists(self.output_dir_tracked):
            os.makedirs(self.output_dir_tracked)

        # --- 只保存一张图片的标志位 (如果需要) ---
        self.raw_saved = False
        self.tracked_saved = False

        # 创建订阅者，订阅两个图像话题
        self.image_raw_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_raw)
        self.image_tracked_sub = rospy.Subscriber("/camera/color/helmet_tracked", Image, self.callback_tracked)
        
        rospy.loginfo("Image saver node started. Subscribing to topics...")
        rospy.loginfo("Saving raw images to: %s", self.output_dir_raw)
        rospy.loginfo("Saving tracked images to: %s", self.output_dir_tracked)

    def callback_raw(self, data):
        # 如果只想保存一张图片，取消下面这行注释
        # if self.raw_saved: return
        
        try:
            # 将ROS图像消息转换为OpenCV图像 (BGR8格式)
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # 创建文件名，使用时间戳确保唯一性
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # 加上毫秒以避免文件名冲突
        milliseconds = int((rospy.Time.now().to_nsec() % 1e9) / 1e6)
        filename = os.path.join(self.output_dir_raw, f"raw_{timestamp}_{milliseconds}.png")

        # 保存图像
        cv2.imwrite(filename, cv_image)
        rospy.loginfo(f"Saved raw image: {filename}")
        
        # 如果只想保存一张图片，取消下面这行注释
        # self.raw_saved = True

    def callback_tracked(self, data):
        # 如果只想保存一张图片，取消下面这行注释
        # if self.tracked_saved: return

        try:
            # 将ROS图像消息转换为OpenCV图像 (BGR8格式)
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # 创建文件名
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        milliseconds = int((rospy.Time.now().to_nsec() % 1e9) / 1e6)
        filename = os.path.join(self.output_dir_tracked, f"tracked_{timestamp}_{milliseconds}.png")
        
        # 保存图像
        cv2.imwrite(filename, cv_image)
        rospy.loginfo(f"Saved tracked image: {filename}")
        
        # 如果只想保存一张图片，取消下面这行注释
        # self.tracked_saved = True

if __name__ == '__main__':
    try:
        saver = ImageSaver()
        # 保持节点运行，直到被关闭
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Image saver node shut down.")
        cv2.destroyAllWindows()