# 文件: ~/Workspace/image_ws/src/image_inverter/image_inverter/helmet_tracker.py

# ========================== 关键修正 - 开始 ==========================
# 在这个文件的最顶部，确保模块搜索路径是正确的
import sys
import os

# 1. 添加 Conda 的 site-packages 路径，确保能找到 torch, transformers 等
#    我们通过运行 python 来动态找到这个路径。
import site
# site.getsitepackages() 返回一个列表，第一个通常是主site-packages
# site_packages_paths = site.getsitepackages()
# for path in site_packages_paths:
#     if path not in sys.path:
#         sys.path.append(path)

image_pipeline_path = os.path.abspath("/home/nvidia/miniconda3/envs/imagepipeline_env/lib/python3.10/site-packages")
if image_pipeline_path not in sys.path:
    sys.path.insert(0, image_pipeline_path)

# --- 现在开始正常的导入 ---
import cv2
import numpy as np
import torch
from enum import Enum, auto

# 2. 添加我们自己的项目代码根目录，确保能找到 sam2, cutie, ImagePipeline
image_pipeline_path = os.path.abspath(os.path.expanduser('~/Workspace/imagepipeline_conda/ImagePipeline'))
if image_pipeline_path not in sys.path:
    sys.path.insert(0, image_pipeline_path)

# 导入VLM相关
from imagepipeline.model_adapters import OpenAIAdapter
#from ImagePipeline.imagepipeline.helmet_detector_base import HelmetDetectorBase # 我们将创建一个基础类
from .helmet_detector_base import HelmetDetectorBase


project_base_path_sam2 = os.path.abspath(os.path.expanduser('~/Workspace/imagepipeline_conda/sam2'))
if project_base_path_sam2 not in sys.path:
    # 把它放在最前面，确保优先搜索我们自己的代码
    sys.path.insert(0, project_base_path_sam2)
# 导入SAM2和CUTIE相关
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

project_base_path_cutie = os.path.abspath(os.path.expanduser('~/Workspace/imagepipeline_conda/Cutie'))
if project_base_path_cutie not in sys.path:
    # 把它放在最前面，确保优先搜索我们自己的代码
    sys.path.insert(0, project_base_path_cutie)
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

from torchvision.transforms.functional import to_tensor
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

# 定义跟踪状态
class TrackingState(Enum):
    SEARCHING = auto()  # 正在使用VLM寻找目标
    TRACKING = auto()   # 正在使用CUTIE跟踪目标

class HelmetTracker:
    """
    一个集成了VLM检测、SAM2分割和CUTIE跟踪的完整流程处理器。
    """
    def __init__(self, sam_checkpoint: str, sam_model_config: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"HelmetTracker using device: {self.device}")

        # --- 1. 初始化 VLM (用于检测) ---
        print("Initializing VLM detector...")
        self.vlm_detector = HelmetDetectorBase() # 使用重构的VLM基础检测器
        print("VLM detector initialized.")
        
        # --- 2. 初始化 SAM2 (用于从bbox生成初始mask) ---
        print("Initializing SAM2 predictor...")
        try:
            if GlobalHydra().is_initialized(): GlobalHydra.instance().clear()
            with initialize(version_base=None, config_path="pkg://sam2"):
                sam_model = build_sam2(sam_model_config, sam_checkpoint, device=self.device)
                self.sam_predictor = SAM2ImagePredictor(sam_model)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM2: {e}")
        print("SAM2 predictor initialized.")

        # --- 3. 初始化 CUTIE (用于跟踪mask) ---
        print("Initializing CUTIE tracker...")
        with torch.no_grad():
            if GlobalHydra().is_initialized(): GlobalHydra.instance().clear()
            cutie_model = get_default_model().to(self.device)
            self.cutie_processor = InferenceCore(cutie_model, cfg=cutie_model.cfg)
        print("CUTIE tracker initialized.")
        
        # --- 4. 状态管理 ---
        self.state = TrackingState.SEARCHING
        self.get_logger().info(f"Initial state: {self.state.name}")

    def get_logger(self):
        # 模拟ROS节点的logger
        import logging
        return logging.getLogger("HelmetTracker")
        
    def reset(self):
        """重置跟踪状态，返回到SEARCHING模式。"""
        self.cutie_processor.clear_memory()
        self.state = TrackingState.SEARCHING
        self.get_logger().info("Tracker has been reset. Returning to SEARCHING state.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像，根据当前状态执行检测或跟踪，并返回可视化结果。
        """
        if self.state == TrackingState.SEARCHING:
            return self._search_and_initialize(frame)
        elif self.state == TrackingState.TRACKING:
            return self._track(frame)
        empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        return frame, empty_mask

    def _search_and_initialize(self, frame: np.ndarray) -> np.ndarray:
        """
        在SEARCHING状态下运行：使用VLM检测，然后用SAM+CUTIE初始化。
        """
        self.get_logger().info("SEARCHING for helmet...")
        empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # 1. 使用VLM获取BBox
        bboxes = self.vlm_detector.detect(frame)
        if not bboxes:
            self.get_logger().info("No helmet detected by VLM.")
            return frame, empty_mask  # 未检测到目标，返回原图

        # VLM很慢，所以我们只处理检测到的第一个目标
        bbox = bboxes[0]
        self.get_logger().info(f"VLM detected helmet at: {bbox}. Initializing tracker...")

        # 2. 使用SAM2从BBox生成初始Mask
        # cv2图像是BGR，SAM需要RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            self.sam_predictor.set_image(rgb_frame)
            masks, scores, _ = self.sam_predictor.predict(box=np.array(bbox), multimask_output=True)
            initial_mask = masks[np.argmax(scores)]

        # 3. 使用初始Mask初始化CUTIE
        # CUTIE需要一个带有对象ID的mask (从1开始)
        initial_mask_with_id = initial_mask.astype(np.uint8) 
        
        # 检查mask是否有效
        if np.sum(initial_mask_with_id) < 50: # 阈值，防止无效的小mask
            self.get_logger().warn("SAM generated a very small or empty mask. Re-searching.")
            # 绘制检测框以供调试
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # 红色表示检测到但初始化失败
            return frame, empty_mask

        mask_tensor = torch.from_numpy(initial_mask_with_id).to(self.device)
        frame_tensor = to_tensor(rgb_frame).to(self.device)
        
        self.cutie_processor.clear_memory()
        self.cutie_processor.step(frame_tensor, mask_tensor, [1]) # 对象ID为1

        # 4. 切换到TRACKING状态
        self.state = TrackingState.TRACKING
        self.get_logger().info(f"Tracker initialized successfully. Switched to {self.state.name} state.")
        
        # 可视化第一帧的结果（bbox + mask）并返回
        vis_frame = self._visualize_mask_and_bbox(frame, initial_mask, bbox)
        return vis_frame, initial_mask.astype(np.uint8) * 255 # 返回可视化帧和二值掩码
    
    def _track(self, frame: np.ndarray) -> np.ndarray:
        """
        在TRACKING状态下运行：使用CUTIE更新mask。
        """
        self.get_logger().info("TRACKING helmet...")
        # #start
        # bboxes = self.vlm_detector.detect(frame)
        # if not bboxes:
        #     self.get_logger().info("No helmet detected by VLM.")
        #     return frame, empty_mask  # 未检测到目标，返回原图
        # #finish
        # # VLM很慢，所以我们只处理检测到的第一个目标
        # bbox = bboxes[0]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = to_tensor(rgb_frame).to(self.device)

        # CUTIE推理
        with torch.no_grad():
            prob = self.cutie_processor.step(frame_tensor)
            tracked_mask = (torch.max(prob, dim=0)[1] > 0).cpu().numpy()

        # 检查是否丢失目标
        if np.sum(tracked_mask) < 50: # 阈值，如果mask太小则认为丢失
            self.get_logger().warn("Tracking lost (mask too small). Resetting tracker.")
            self.reset()
            # 在丢失的帧上不做任何绘制，下一帧将重新搜索
            empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            return frame, empty_mask

        # 可视化跟踪的mask并返回
        vis_frame = self._visualize_mask_and_bbox(frame, tracked_mask)
        # vis_frame = self._visualize_mask_and_bbox(frame, tracked_mask, bbox)
        return vis_frame, tracked_mask.astype(np.uint8) * 255 # 返回可视化帧和二值掩码

    def _visualize_mask_and_bbox(self, frame: np.ndarray, mask: np.ndarray, bbox: list = None) -> np.ndarray:
        """在图像上绘制mask和可选的bbox。"""
        # 创建一个彩色的叠加层
        overlay = frame.copy()
        color = (0, 255, 0) # 绿色
        alpha = 0.4
        
        # 将mask应用到叠加层
        overlay[mask > 0] = color
        
        # 将叠加层与原图混合
        vis_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
        return vis_frame