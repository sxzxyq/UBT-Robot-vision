# UBT-Robot-vision 🤖

**基于 ROS 2 的天工人形机器人视觉感知系统**

本项目是一个视觉感知系统，主要用于安全合规检测（如安全帽佩戴检测）。

## 📂 项目结构

ROS 工作空间和算法库如下：

```text
UBT-Robot-vision/
├── image_ws/               # [ROS 2] 视觉服务与图像处理工作空间
│   ├── src/image_inverter/ # 核心视觉逻辑
│   └── src/image_inverter_interfaces/ # 自定义服务接口 (TriggerVerification)
├── vision_ws/              # [ROS 2] 测试脚本
│   ├── src/Cutie/          # 视频对象分割与跟踪算法
│   ├── src/sam2/           # Segment Anything Model 2
│   └── src/ImagePipeline/  # 图像处理管道
├── imagepipeline_conda/    # [Python] 用于 Conda 环境
├── pcd_store/              # 点云数据存储
├── requirements.txt        # Python 依赖列表
├── run_all.sh              # 一键启动脚本
└── .gitignore              # Git 忽略配置
```

## 🚀 功能特性

*   **视觉感知**：
    *   集成 **Qwen-VL** (或类似 VLM) 进行语义理解与合规验证（如：是否佩戴安全帽）。
    *   使用 **SAM2/Cutie** 实现实时的目标分割与跟踪。
    *   发布过滤后的目标坐标 (`/helmet/position_filtered_transformed`)。

## 🛠️ 环境依赖

*   **硬件**：Nvidia Jetson Orin / Ubuntu PC
*   **系统**：Ubuntu 22.04
*   **ROS 版本**：ROS 2 Humble
*   **Python**：Python 3.10+ (建议使用 Conda 管理深度学习环境)

## 📦 安装与编译

### 1. 克隆仓库
```bash
git clone https://github.com/sxzxyq/UBT-Robot-vision.git
cd UBT-Robot-vision
```

### 2. 安装 Python 依赖
建议在 Conda 环境中安装深度学习依赖，在系统环境中安装 ROS 依赖。
```bash
pip install -r requirements.txt
```

### 3. 编译 ROS工作空间

**编译 `image_ws` :**
```bash
cd image_ws
colcon build --symlink-install
source install/setup.bash
```

## 运行
*   1.启动机器人的相机节点
*   2.激活虚拟环境
*   3.执行一键启动脚本
