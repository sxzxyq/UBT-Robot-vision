import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import numpy as np

class KalmanFilterNode(Node):
    """
    订阅一个PointStamped话题，对其位置数据应用卡尔曼滤波器，
    并发布滤波后的PointStamped。
    """
    def __init__(self):
        super().__init__('kalman_filter_node')
        self.get_logger().info('Initializing Kalman Filter Node...')

        # === 1. 参数声明 ===
        # 允许从launch文件或命令行调整这些关键参数
        self.declare_parameter('process_noise_std', 0.05) # 过程噪声标准差，假设速度变化不大
        self.declare_parameter('measurement_noise_std', 0.1) # 测量噪声标准差，根据传感器实际情况调整

        # === 2. 订阅与发布 ===
        self.subscription = self.create_subscription(
            PointStamped,
            '/helmet/position',  # 订阅原始位置数据
            self.position_callback,
            10)
        self.publisher = self.create_publisher(
            PointStamped,
            '/helmet/position_filtered',  # 发布滤波后的位置数据
            10)

        # === 3. 卡尔曼滤波器状态初始化 ===
        self.is_initialized = False
        self.last_update_time = None

        # 状态向量 [x, y, z, vx, vy, vz]'. 6x1.
        self.state = np.zeros((6, 1)) 
        
        # 状态协方差矩阵 P. 6x6. 初始不确定性很大.
        self.covariance = np.eye(6) * 500.

        # 状态转移矩阵 F. 6x6. 后面会动态更新dt.
        self.F = np.eye(6)

        # 测量矩阵 H. 3x6. 我们只能测量到位置.
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # 过程噪声协方差矩阵 Q. 6x6.
        # 代表了我们对“匀速”模型的信任程度。值越大，越不信任模型，能更快适应加速度变化。
        process_noise_std = self.get_parameter('process_noise_std').get_parameter_value().double_value
        q_val = process_noise_std ** 2
        self.Q = np.diag([0, 0, 0, q_val, q_val, q_val]) # 噪声主要在速度上

        # 测量噪声协方差矩阵 R. 3x3.
        # 代表了我们对传感器的信任程度。值越大，说明传感器越不准。
        measurement_noise_std = self.get_parameter('measurement_noise_std').get_parameter_value().double_value
        self.R = np.eye(3) * (measurement_noise_std ** 2)

        self.get_logger().info('Kalman Filter Node started.')
        self.get_logger().info(f"Process noise std: {process_noise_std}, Measurement noise std: {measurement_noise_std}")


    def position_callback(self, msg):
        current_time = self.get_clock().now()

        # --- 初始化步骤 ---
        if not self.is_initialized:
            self.state[0] = msg.point.x
            self.state[1] = msg.point.y
            self.state[2] = msg.point.z
            # 初始速度为0
            self.last_update_time = current_time
            self.is_initialized = True
            self.get_logger().info('Kalman filter initialized with first measurement.')
            return

        # --- 滤波循环 ---
        # 计算时间差 dt
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time
        
        # 1. 预测 (Predict)
        # 更新状态转移矩阵 F
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # 预测状态和协方差
        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q

        # 2. 更新 (Update)
        # 从消息中获取测量值 z
        measurement = np.array([[msg.point.x], [msg.point.y], [msg.point.z]])

        # 计算卡尔曼增益 K
        measurement_residual = measurement - self.H @ predicted_state
        residual_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(residual_covariance)

        # 更新状态和协方差
        self.state = predicted_state + kalman_gain @ measurement_residual
        self.covariance = (np.eye(6) - kalman_gain @ self.H) @ predicted_covariance

        # --- 发布滤波后的结果 ---
        filtered_msg = PointStamped()
        filtered_msg.header = msg.header # 使用与输入消息相同的时间戳和坐标系
        filtered_msg.point.x = self.state[0, 0]
        filtered_msg.point.y = self.state[1, 0]
        filtered_msg.point.z = self.state[2, 0]
        self.publisher.publish(filtered_msg)
        
        # (可选) 打印日志用于调试
        # self.get_logger().info(f"Filtered pos: [{self.state[0,0]:.3f}, {self.state[1,0]:.3f}, {self.state[2,0]:.3f}]")

def main(args=None):
    rclpy.init(args=args)
    kalman_filter_node = KalmanFilterNode()
    try:
        rclpy.spin(kalman_filter_node)
    except KeyboardInterrupt:
        pass
    finally:
        kalman_filter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()