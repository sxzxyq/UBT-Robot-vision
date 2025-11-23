#!/bin/bash
set -e

echo "==> Creating clean ROS+Conda environment..."

# --- 步骤 1: 提取Conda路径 ---
# 我们仍然需要这样做来找到torch等库
_CONDA_BASE=$(conda info --base)
source "$_CONDA_BASE/etc/profile.d/conda.sh"
conda activate imagepipeline_env
CONDA_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
conda deactivate

# --- 步骤 2: Source ROS环境 ---
source ~/Workspace/image_ws/install/setup.bash

# --- 步骤 3: 构建最终的 PYTHONPATH ---
# 这个 PYTHONPATH 是为 系统Python (3.10) 准备的
export PYTHONPATH="$CONDA_SITE_PACKAGES:/home/nvidia/Workspace/imagepipeline_conda:$PYTHONPATH"

# --- 步骤 4: 使用正确的Python解释器运行脚本 ---
# 这是最最关键的修改！
# 我们不再使用模糊的 'python3'，而是用绝对路径指定ROS兼容的解释器。
echo "==> Environment ready. Running save_pcd.py with /usr/bin/python3 ..."
/usr/bin/python3 save_pcd.py