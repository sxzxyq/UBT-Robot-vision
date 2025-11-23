#!/bin/bash

# =============================================================================
#           终极启动脚本 (V3 - 包含 ABI 冲突解决方案)
# =============================================================================
set -e

# --- 步骤 1: 激活 Conda，但只为了获取 Python 路径 ---
echo "==> Activating Conda environment to get Python paths..."
_CONDA_BASE=$(conda info --base)
source "$_CONDA_BASE/etc/profile.d/conda.sh"
conda activate imagepipeline_env

# 从激活的环境中，提取出我们唯一需要的两个路径
CONDA_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
YOUR_PROJECT_ROOT="/home/nvidia/Workspace/imagepipeline_conda"

# --- 步骤 2: 立即停用 Conda，回到一个干净的 Shell ---
# 这是关键！我们抛弃掉 Conda 带来的所有底层库路径 (LD_LIBRARY_PATH)
echo "==> Deactivating Conda to clean the environment..."
conda deactivate

# --- 步骤 3: 在干净的 Shell 中，构建我们的“黄金”PYTHONPATH ---
# 首先加载 ROS 的基础路径
echo "==> Sourcing ROS 2 workspace..."
source "$(pwd)/install/setup.bash"

# 然后，将我们从 Conda 中提取的、无害的 Python 路径注入进去
echo "==> Injecting Conda Python paths into PYTHONPATH..."
export PYTHONPATH="$CONDA_SITE_PACKAGES:$YOUR_PROJECT_ROOT:$PYTHONPATH"

# --- 步骤 4: 运行！ ---
# 现在我们有了一个完美的环境：
# - 底层C++库 (LD_LIBRARY_PATH) 来自于 ROS 和系统，是兼容的。
# - Python 库 (PYTHONPATH) 来自于 Conda 和我们自己，是我们需要的。
echo "==> Final Environment:"
echo "    WHICH PYTHON: $(which python)" # 应该是 /usr/bin/python3
echo "    PYTHONPATH contains Conda site-packages: $(echo $PYTHONPATH | grep -q $CONDA_SITE_PACKAGES && echo Yes || echo No)"
echo "    LD_LIBRARY_PATH does NOT contain Conda lib: $(echo $LD_LIBRARY_PATH | grep -q conda && echo No || echo Yes)"
echo "-----------------------------------------------------"

echo "==> Launching all nodes..."
# 你可以把 save_pcd 也加到 launch 文件里，或者单独运行
LAUNCH_FILE="/tmp/full_system_launch.py"

echo "==> Generating temporary launch file at $LAUNCH_FILE"
cat > "$LAUNCH_FILE" << EOL
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 启动头盔跟踪器节点
        Node(
            package='image_inverter',
            executable='helmet_tracker',
            name='helmet_tracker_node',
            output='log'
        ),
        # 启动深度图转点云节点
        Node(
            package='image_inverter',
            executable='depth_to_pointcloud',
            name='depth_to_pointcloud_node',
            output='log'
        ),
        # 启动融合节点
        Node(
            package='image_inverter',
            executable='fusion',
            name='fusion_node',
            output='log'
        ),
        # 启动卡尔曼滤波节点
        Node(
            package='image_inverter',
            executable='kalman_filter_node',
            name='kalman_filter_node',
            output='log'
        ),
        # 启动辅助验证节点
        Node(
            package='image_inverter',
            executable='secondary_verifier',
            name='secondary_verifier_node',
            output='screen'
        ),
        # 启动数量验证节点
        Node(
            package='image_inverter',
            executable='object_counter',
            name='object_counter_node',
            output='screen'
        )
    ])
EOL

echo "==> Launching all nodes..."
# 使用我们创建的launch文件来启动所有节点
ros2 launch "$LAUNCH_FILE"