import torch
import json_repair
import cv2
import numpy as np
import base64
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from torchvision.transforms.functional import to_tensor
from io import BytesIO
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

from utils import visualize, compute_mask_centers, visualize_centers
from model_adapters import QwenVLAdapter,OpenAIAdapter

import os
from tqdm import tqdm
import re  # <-- 新增: 导入正则表达式库用于解析数字
import json # <-- 新增: 导入JSON库用于保存结果

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Realsense library not available. Skipping RealsenseCamera functionality.")


class ImagePipeline:
    def __init__(
        self, sam_checkpoint: str, sam_model_config: str, cutie_model, vl_adapter
    ):
        """
        多目标分割与跟踪管道（集成视觉语言模型）

        参数:
        sam_checkpoint: SAM模型权重路径
        sam_model_config: SAM模型类型配置 (i.e. "configs/sam2.1/sam2.1_hiera_l.yaml")
        cutie_model: 预加载的CUTIE模型实例
        vl_adapter: 视觉语言模型适配器实例（QwenVLAdapter）
        """
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        try:
            # 清除Hydra状态
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()

            # 重新初始化Hydra，使用相对路径
            with initialize(version_base=None, config_path="pkg://sam2"):
                self.predictor = SAM2ImagePredictor(
                    build_sam2(sam_model_config, sam_checkpoint, device=self.device)
                )
        except Exception as e:
            print(f"SAM2 initialization error: {e}")
            # 备用方案：尝试不同的相对路径
            try:
                GlobalHydra.instance().clear()
                with initialize(version_base=None, config_path="../../sam2/sam2"):
                    self.predictor = SAM2ImagePredictor(
                        build_sam2(sam_model_config, sam_checkpoint, device=self.device)
                    )
            except Exception as e2:
                print(f"Backup SAM2 initialization also failed: {e2}")
                raise

        # 初始化CUTIE - 确保模型在正确设备上
        self.cutie = cutie_model.to(self.device)
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = -1

        # 集成视觉语言模型
        self.vl_adapter = vl_adapter

        # 状态跟踪
        self.current_objects = []
        self.is_initialized = False

    def _image_to_base64(self, image: np.ndarray) -> str:
        """将numpy图像转换为base64编码字符串"""
        pil_img = Image.fromarray(image)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_bbox_from_vl(self, frame: np.ndarray, instruction: str) -> list:
        """
        使用视觉语言模型生成目标边界框

        参数:
        frame: RGB格式的输入图像 [H, W, 3]
        instruction: 自然语言指令（如"the red cup on the left"）

        返回:
        bboxes: 边界框列表 [[x1,y1,x2,y2], ...]
        """
        # 转换图像格式
        base64_image = self._image_to_base64(frame)

        # 构建VL模型输入
        prompt = (
            f"Analyze the image and identify ALL objects matching: {instruction}.\n"
            "Return bboxes for ALL matching objects in this format:\n"
            '[{"bbox_2d": [x1,y1,x2,y2], "label": "..."}, ...]'
        )

        print(f"vl user prompt:\n{prompt}")

        # 生成响应
        input_data = self.vl_adapter.prepare_input(
            text=prompt, image_url=f"data:image/jpeg;base64,{base64_image}"
        )
        response, _ = self.vl_adapter.generate_response(input_data, max_tokens=512)

        print(f"response from vl: {response}")

        # 解析响应
        try:
            # 提取JSON部分
            json_str = response[response.find("[") : response.rfind("]") + 1]
            bbox_list = json_repair.loads(json_str)

            # 验证bbox格式
            valid_bboxes = []
            for item in bbox_list:
                bbox = item.get("bbox_2d", [])
                if len(bbox) == 4 and all(
                    0 <= v <= frame.shape[1] if i % 2 == 0 else 0 <= v <= frame.shape[0]
                    for i, v in enumerate(bbox)
                ):
                    valid_bboxes.append(bbox)
            return valid_bboxes
        except Exception as e:
            raise RuntimeError(f"Failed to parse VL model response: {str(e)}")

    def initialize_with_instruction(
        self, frame: np.ndarray, instruction: str, return_bbox: bool = False
    ) -> tuple[np.ndarray, list | None]:
        """
        端到端初始化流程：VL生成bbox -> SAM分割 -> CUTIE初始化

        参数:
        frame: RGB格式的输入图像
        instruction: 自然语言指令

        返回:
        combined_mask: 组合后的多目标mask
        """
        # Step 1: 通过VL模型获取bbox
        bboxes = self.get_bbox_from_vl(frame, instruction)
        if not bboxes:
            raise ValueError("No valid bounding boxes detected by VL model")

        # Step 2: SAM生成mask
        return self.initialize_masks(frame, bboxes), None if not return_bbox else bboxes

    def initialize_masks(self, frame: np.ndarray, bboxes: list) -> np.ndarray:
        """
        初始化多目标分割

        参数:
        frame: RGB格式的输入图像 [H, W, 3]
        bboxes: 多个目标的边界框列表 [[x1, y1, x2, y2], ...]

        返回:
        combined_mask: 组合后的多目标mask，每个目标用不同整数ID表示
        """
        # 转换颜色空间并设置SAM图像
        rgb_frame = frame
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb_frame)

            # 生成并组合多个目标的mask
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            object_ids = []
            for obj_idx, bbox in enumerate(bboxes):
                # SAM预测最佳mask
                masks, scores, _ = self.predictor.predict(
                    box=np.array(bbox), multimask_output=True
                )
                best_mask = masks[np.argmax(scores)]

                # 确保mask是布尔类型
                best_mask = best_mask.astype(bool)

                # 分配唯一对象ID (从1开始)
                obj_id = obj_idx + 1
                combined_mask[best_mask] = obj_id
                object_ids.append(obj_id)

        # 初始化CUTIE处理器
        mask_tensor = torch.from_numpy(combined_mask).to(self.device)
        self.processor.clear_memory()
        self.processor.step(
            to_tensor(rgb_frame).to(self.device), mask_tensor, object_ids
        )

        # 更新状态
        self.current_objects = object_ids
        self.is_initialized = True

        return combined_mask

    def update_masks(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        更新多目标跟踪结果

        参数:
        frame: RGB格式的新帧 [H, W, 3]

        返回:
        list: 每个目标的二值mask列表 [mask1, mask2, ...]
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_masks first.")

        # 准备输入数据
        rgb_frame = frame
        image_tensor = to_tensor(rgb_frame).to(self.device)

        # CUTIE推理
        with torch.no_grad():
            output_prob = self.processor.step(image_tensor)
            current_mask = self.processor.output_prob_to_mask(output_prob)
            current_mask_np = current_mask.cpu().numpy().astype(np.uint8)

        # 分离各个目标的mask
        return current_mask_np, [
            (current_mask_np == obj_id) for obj_id in self.current_objects
        ]

    def reset(self):
        """重置管道状态"""
        self.processor.clear_memory()
        self.current_objects = []
        self.is_initialized = False

    def add_object(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        动态添加新目标到现有跟踪

        参数:
        frame: RGB格式的当前帧
        bbox: 新目标的边界框 [x1, y1, x2, y2]

        返回:
        new_mask: 新目标的单独mask
        """
        # 生成新目标mask
        rgb_frame = frame[..., ::-1]
        self.predictor.set_image(rgb_frame)
        masks, scores, _ = self.predictor.predict(
            box=np.array(bbox), multimask_output=True
        )
        new_mask = masks[np.argmax(scores)]

        # 分配新ID
        new_id = max(self.current_objects) + 1 if self.current_objects else 1
        new_mask_tensor = torch.from_numpy(new_mask.astype(np.uint8) * new_id).to(
            self.device
        )

        # 合并到现有mask
        combined_mask = self.processor.output_prob_to_mask(self.processor.prob)
        combined_mask = torch.where(new_mask_tensor > 0, new_mask_tensor, combined_mask)

        # 更新处理器状态
        self.current_objects.append(new_id)
        self.processor.step(image_tensor, combined_mask, self.current_objects)

        return new_mask


class RealsenseCamera:
    # TODO havent been tested
    def __init__(self):
        device_id = "203522250675"
        self.pipeline = rs.pipeline()  # type: ignore
        self.config = rs.config()  # type: ignore
        print(f"[RealsenseCamera]: config: {self.config}")
        self.config.enable_device(device_id)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # type: ignore

    def __enter__(self):
        self.pipeline.start(self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipeline.stop()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        return frame


def main_rs(
    frame_callback=None,
    save_output: bool = False,
    save_video: bool = False,
    disable_vlm=False,
    video_source="camera",
    video_path=None,
):
    print("start!")
    # 参数验证
    if video_source == "video" and video_path is None:
        raise ValueError("video_path is required when video_source is 'video'")

    # 初始化所有模型
    sam_checkpoint = "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # 初始化视频源
    camera = None
    cap = None
    try:
        if video_source == "camera":
            camera = RealsenseCamera()
            camera.__enter__()  # 手动进入上下文
            frame = camera.get_frame()
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read first frame from video")

        print(frame.shape)

        # 视频保存初始化
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                "output.mp4", fourcc, 30.0, (frame.shape[1], frame.shape[0])
            )
        if save_output:
            fourccm = cv2.VideoWriter_fourcc(*"mp4v")
            maskedvideo_writer = cv2.VideoWriter(
                "outputmask.mp4",
                fourccm,
                30.0,
                (frame.shape[1], frame.shape[0]),
            )
        if not disable_vlm:
            cutie_model = get_default_model()
            vl_adapter = QwenVLAdapter(
                model_path="/data/model/Qwen/Qwen2.5-VL-3B-Instruct"
            )

            pipeline = ImagePipeline(
                sam_checkpoint, sam_model_config, cutie_model, vl_adapter
            )

        # 初始化标志
        initialized = False
        try:
            i = 0
            while True:
                i += 1
                # 获取帧
                if video_source == "camera":
                    frame = camera.get_frame()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break  # 视频结束退出循环

                rgb_frame = frame[:, :, ::-1].copy()
                if not disable_vlm:
                    if not initialized:
                        # 使用第一帧进行初始化
                        combined_mask, bboxes = pipeline.initialize_with_instruction(
                            frame=rgb_frame,
                            instruction="lemon",
                            return_bbox=True,
                        )
                        initialized = True
                        centers = compute_mask_centers(combined_mask, "centroid")
                        updated_mask = combined_mask
                    else:
                        # 更新掩码
                        updated_mask, _ = pipeline.update_masks(rgb_frame)
                        centers = compute_mask_centers(updated_mask, "centroid")
                        print(centers)
                    # 实时可视化
                    if False:
                        vis_frame = rgb_frame.copy()
                        visualize(
                            vis_frame,
                            bboxes=bboxes if not initialized else None,
                            mask=updated_mask,
                            save_path=f"masked_{i}.png",
                        )
                        visualize_centers(vis_frame, centers=centers)
                    if save_output:
                        print("save maskframe")
                        over_lay = add_mask(updated_mask, rgb_frame)
                        maskedvideo_writer.write(over_lay)
                # import time
                # time.sleep(0.1)
                # 视频保存逻辑
                if save_video:
                    print("save frame")
                    video_writer.write(rgb_frame)
                    # 视频保存逻辑

                if not disable_vlm and frame_callback:
                    frame_callback(centers)
                    # pass
        except KeyboardInterrupt:
            print("STOPPED")
        finally:
            if save_video and video_writer is not None:
                video_writer.release()
    finally:
        # 资源清理
        if video_source == "camera" and camera is not None:
            print("stop camera")
            camera.__exit__(None, None, None)  # 手动退出上下文
        elif cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main_rs_iter(
    save_output: bool = False,
    save_video: bool = False,
    disable_vlm=False,
    video_source="camera",
    video_path=None,
):
    print("start!")
    # 参数验证
    if video_source == "video" and video_path is None:
        raise ValueError("video_path is required when video_source is 'video'")

    # 初始化所有模型
    sam_checkpoint = "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # 初始化视频源
    camera = None
    cap = None
    try:
        if video_source == "camera":
            camera = RealsenseCamera()
            camera.__enter__()  # 手动进入上下文
            frame = camera.get_frame()
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read first frame from video")

        print(frame.shape)

        # 视频保存初始化
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                "output.mp4", fourcc, 30.0, (frame.shape[1], frame.shape[0])
            )
        if save_output:
            fourccm = cv2.VideoWriter_fourcc(*"mp4v")
            maskedvideo_writer = cv2.VideoWriter(
                "outputmask.mp4",
                fourccm,
                30.0,
                (frame.shape[1], frame.shape[0]),
            )
        if not disable_vlm:
            cutie_model = get_default_model()
            vl_adapter = QwenVLAdapter(
                model_path="/data/model/Qwen/Qwen2.5-VL-3B-Instruct"
            )
            pipeline = ImagePipeline(
                sam_checkpoint, sam_model_type, cutie_model, vl_adapter
            )

        # 初始化标志
        initialized = False
        try:
            i = 0
            while True:
                i += 1
                # 获取帧
                if video_source == "camera":
                    frame = camera.get_frame()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break  # 视频结束退出循环

                rgb_frame = frame[:, :, ::-1].copy()
                if not disable_vlm:
                    if not initialized:
                        # 使用第一帧进行初始化
                        combined_mask, bboxes = pipeline.initialize_with_instruction(
                            frame=rgb_frame,
                            instruction="Strawberry",
                            return_bbox=True,
                        )
                        initialized = True
                        centers = compute_mask_centers(combined_mask, "centroid")
                        updated_mask = combined_mask
                    else:
                        # 更新掩码
                        updated_mask, _ = pipeline.update_masks(rgb_frame)
                        centers = compute_mask_centers(updated_mask, "centroid")

                    # 实时可视化
                    if i % 10 == 0:
                        vis_frame = rgb_frame.copy()
                        visualize(
                            vis_frame,
                            bboxes=bboxes if not initialized else None,
                            mask=updated_mask,
                            save_path=f"masked_{i}.png",
                        )
                        visualize_centers(vis_frame, centers=centers)

                # 视频保存逻辑
                if save_video:
                    print("save frame")
                    video_writer.write(rgb_frame)
                    # 视频保存逻辑
                if save_output:
                    print("save maskframe")
                    maskedvideo_writer.write(rgb_frame)
                if not disable_vlm:
                    yield centers
        except KeyboardInterrupt:
            print("STOPPED")
        finally:
            if save_video and video_writer is not None:
                video_writer.release()
    finally:
        # 资源清理
        if video_source == "camera" and camera is not None:
            camera.__exit__(None, None, None)  # 手动退出上下文
        elif cap is not None:
            cap.release()
        cv2.destroyAllWindows()

def main_qwen_demo():
    """
    独立测试 VLM 功能：输入图像和提示词，输出识别框。
    此函数不加载 SAM 和 CUTIE 模型。
    """
    print("--- Starting OpenAI BBox-Only Demo ---")

    # 1. 初始化 OpenAI 适配器
    print("Initializing OpenAI Adapter...")
    try:
        # base_url 需要指向您的本地服务器的 API 入口点，通常包含 /v1 后缀
        vl_adapter = OpenAIAdapter(
            api_key="EMPTY",  # 对于本地模型通常不需要，但库要求非空
            base_url="http://10.134.159.154:8000/v1",
            model_name=None  # 设置为 None 以自动检测模型
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI Adapter: {e}")
        return

    # 2. 加载图像
    image_path = "./imagepipeline/test_images/image_test.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from path: {image_path}")
        return
    
    # 将 cv2 读取的 BGR 图像转换为 VLM 需要的 RGB 格式
    rgb_frame = frame[:, :, ::-1].copy()

    # 3. 准备 VLM 输入 (逻辑从 ImagePipeline.get_bbox_from_vl 提取)
    
    # 辅助函数：将 numpy 图像转换为 base64 字符串
    def _image_to_base64(image: np.ndarray) -> str:
        pil_img = Image.fromarray(image)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    base64_image = _image_to_base64(rgb_frame)
    
    # prompt_v0 
    # instruction = "the safety helmet"
    # prompt = (
    #     f"Analyze the image and identify ALL objects matching: {instruction}.\n"
    #     "Return bboxes for ALL matching objects in this format:\n"
    #     '[{"bbox_2d": [x1,y1,x2,y2], "label": "..."}, ...]'
    # )
    
    #prompt_v1
    instruction = "the safety helmet on the worker's head"
    prompt = (
        f"Your task is to act as a precise object detector.\n"
        f"In the image, find the exact location of the object described as: '{instruction}'.\n"
        f"The bounding box must be as tight as possible around ONLY the helmet itself.\n"
        f"It is critical that you DO NOT include the person's head, face, or body in the box.\n"
        f"Return bboxes for ALL matching objects in this format:\n"
        '[{"bbox_2d": [x1,y1,x2,y2], "label": "safety helmet"}, ...]'
    )
    

    print(f"\n--- Running inference with instruction: '{instruction}' ---")
    print(f"Prompt sent to model:\n{prompt}")

    # 4. 调用 VLM 并获取响应
    try:
        input_data = vl_adapter.prepare_input(
            text=prompt, image_url=f"data:image/jpeg;base64,{base64_image}"
        )
        response, _ = vl_adapter.generate_response(input_data, max_tokens=1024)
        print(f"\nRaw response from VL model:\n{response}")
    except Exception as e:
        print(f"\nError during API call to the VL model: {e}")
        return

    # 5. 解析响应以提取边界框
    try:
        # 尝试从响应中提取 JSON 字符串
        json_str = response[response.find("[") : response.rfind("]") + 1]
        if not json_str: # 如果没找到[],尝试从 markdown 代码块中提取
             if "```json" in response:
                 json_str = response.split("```json")[1].split("```")[0].strip()
        
        bbox_list = json_repair.loads(json_str)

        # 验证 bbox 格式并转换为整数
        valid_bboxes = []
        for item in bbox_list:
            bbox = item.get("bbox_2d", [])
            if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                # 确保坐标在图像范围内
                x1, y1, x2, y2 = bbox
                h, w, _ = frame.shape
                x1 = max(0, min(w - 1, int(x1)))
                y1 = max(0, min(h - 1, int(y1)))
                x2 = max(0, min(w - 1, int(x2)))
                y2 = max(0, min(h - 1, int(y2)))
                valid_bboxes.append([x1, y1, x2, y2])
            else:
                print(f"Skipping invalid bbox item: {item}")
        
        print(f"\nSuccessfully parsed {len(valid_bboxes)} bounding box(es): {valid_bboxes}")

        if not valid_bboxes:
            print("Warning: No valid bounding boxes were found in the response.")
            return

    except Exception as e:
        print(f"\nFailed to parse VL model response: {str(e)}")
        return

    # 6. 可视化结果
    print("Visualizing results...")
    vis_frame = frame.copy()  # 在原始 BGR 图像上绘制
    visualize(
        vis_frame,
        bboxes=valid_bboxes,
        mask=None,  # 没有掩码
        save_path="./imagepipeline/test_qwen/openai_demo_bbox_only_output.png"
    )
    print("Visualization saved to 'openai_demo_bbox_only_output.png'")    

def main_demo():
    # 初始化所有模型
    #sam_checkpoint = "thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam_checkpoint = "../../sam2/checkpoints/sam2.1_hiera_large.pt"
    #sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_model_config = "../../sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    #vl_adapter = QwenVLAdapter(model_path="/quick_data/model_qwen2.5_vl_7b_instruct")
    vl_adapter = QwenVLAdapter(model_path="Qwen/Qwen2.5-VL-7B-Instruct")
    print("vl_adapter initialized")
    with torch.no_grad():
        try:
            cutie_model = get_default_model()
        except ValueError as e:
            if "GlobalHydra is already initialized" in str(e):
                GlobalHydra.instance().clear()
                cutie_model = get_default_model()
            else:
                raise
        print("Cuite model loaded successfully")
        # 创建增强版管道
        pipeline = ImagePipeline(
            sam_checkpoint, sam_model_config, cutie_model, vl_adapter
        )
        print("Pipeline initialized successfully")
        # 通过自然语言指令初始化
        frame = cv2.imread(
            "logs/20250530_005948/1_0_monitor_assemble_object_input_1.png"
        )
        rgb_frame = frame[:, :, ::-1].copy()
        combined_mask, bboxes = pipeline.initialize_with_instruction(
            frame=frame,
            instruction="robot arm",  # 自然语言指令
            return_bbox=True,
        )

        # 可视化结果
        visualize(frame, bboxes=bboxes, mask=combined_mask)
        centers = compute_mask_centers(combined_mask, "centroid")
        visualize_centers(frame, centers=centers)
        # 后续跟踪流程...
        for _ in range(10):
            new_frame = rgb_frame  # 获取新帧
            updated_mask, _ = pipeline.update_masks(new_frame)
            centers = compute_mask_centers(updated_mask, "centroid")
            visualize(new_frame, mask=updated_mask)
            visualize_centers(new_frame, centers=centers)

def _image_to_base64(image: np.ndarray) -> str:
    """将 numpy 图像 (RGB) 转换为 base64 编码字符串"""
    pil_img = Image.fromarray(image)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- 核心处理逻辑：处理单张图片 ---
# def get_bboxes_for_image(vl_adapter, image_path: str, instruction: str) -> list:
#     """
#     使用 VLM 为单张图片生成边界框。

#     参数:
#     - vl_adapter: 已初始化的 OpenAIAdapter 实例。
#     - image_path: 输入图片的路径。
#     - instruction: 用户的自然语言指令。

#     返回:
#     - 边界框列表 [[x1, y1, x2, y2], ...]，如果失败则返回空列表。
#     """
#     # 1. 加载和准备图像
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print(f"Warning: Could not read image from path: {image_path}")
#         return []
    
#     rgb_frame = frame[:, :, ::-1].copy()
#     base64_image = _image_to_base64(rgb_frame)

#     # 2. 构建优化的提示词 (使用方案一)
#     prompt = (
#         f"Your task is to act as a precise object detector.\n"
#         f"In the image, find the exact location of the object described as: '{instruction}'.\n"
#         f"The bounding box must be as tight as possible around ONLY the object itself.\n"
#         f"It is critical that you DO NOT include surrounding elements or larger objects it is part of.\n"
#         f"Return bboxes for ALL matching objects in this format:\n"
#         '[{"bbox_2d": [x1,y1,x2,y2], "label": "safety helmet"}, ...]'
#     )

    
#     # 3. 调用 VLM 并获取响应
#     try:
#         input_data = vl_adapter.prepare_input(
#             text=prompt, image_url=f"data:image/jpeg;base64,{base64_image}"
#         )
#         response, _ = vl_adapter.generate_response(input_data, max_tokens=1024)
#     except Exception as e:
#         print(f"\nError during API call for image {os.path.basename(image_path)}: {e}")
#         return []

#     # 4. 解析响应以提取边界框
#     try:
#         json_str = response[response.find("[") : response.rfind("]") + 1]
#         if not json_str and "```json" in response:
#             json_str = response.split("```json")[1].split("```")[0].strip()
        
#         bbox_list = json_repair.loads(json_str)

#         valid_bboxes = []
#         for item in bbox_list:
#             bbox = item.get("bbox_2d", [])
#             if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
#                 h, w, _ = frame.shape
#                 x1, y1, x2, y2 = bbox
#                 x1, y1 = max(0, int(x1)), max(0, int(y1))
#                 x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
#                 valid_bboxes.append([x1, y1, x2, y2])
        
#         if not valid_bboxes:
#             print(f"Warning: No valid bboxes found for {os.path.basename(image_path)}. Raw response: {response[:200]}...")

#         return valid_bboxes
#     except Exception as e:
#         print(f"\nFailed to parse response for {os.path.basename(image_path)}: {e}")
#         print(f"Raw response was: {response}")
#         return []



# --- 核心处理逻辑：处理单张图片 (添加了图像缩放) ---
def get_bboxes_for_image(vl_adapter, image_path: str, instruction: str, max_image_size: int = 768) -> list:
    """
    使用 VLM 为单张图片生成边界框。
    增加了图像缩放功能以避免超出模型上下文长度限制。

    参数:
    - vl_adapter: 已初始化的 OpenAIAdapter 实例。
    - image_path: 输入图片的路径。
    - instruction: 用户的自然语言指令。
    - max_image_size: 图像最长边的最大像素值。
    """
    # 1. 加载图像
    original_frame = cv2.imread(image_path)
    if original_frame is None:
        print(f"Warning: Could not read image from path: {image_path}")
        return []

    # --- ↓↓↓ 新增的图像缩放逻辑 ↓↓↓ ---
    h, w = original_frame.shape[:2]
    original_size = (w, h) # 保存原始尺寸 (width, height) 以便后续恢复坐标

    if max(h, w) > max_image_size:
        scale = max_image_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_for_model = cv2.resize(original_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resize_scale = (w / new_w, h / new_h) # (scale_w, scale_h)
        print(f"Resized image {os.path.basename(image_path)} from {w}x{h} to {new_w}x{new_h}")
    else:
        frame_for_model = original_frame
        resize_scale = (1.0, 1.0) # No resize needed
    # --- ↑↑↑ 缩放逻辑结束 ↑↑↑ ---

    # 使用缩放后的图像进行后续处理
    rgb_frame = frame_for_model[:, :, ::-1].copy()
    base64_image = _image_to_base64(rgb_frame)

    # 2. 构建提示词 (不变)
    prompt = (
        f"Your task is to act as a precise object detector.\n"
        f"In the image, find the exact location of the object described as: '{instruction}'.\n"
        f"The bounding box must be as tight as possible around ONLY the object itself.\n"
        f"It is critical that you DO NOT include surrounding elements or larger objects it is part of.\n"
        f"Return bboxes for ALL matching objects in this format:\n"
        f'[{{ "bbox_2d": [x1,y1,x2,y2], "label": "{instruction}" }} , ...]'
        f"if no {instruction} existed in the image, just return 'NO'"
    )
    
    # 3. 调用 VLM (不变)
    try:
        input_data = vl_adapter.prepare_input(
            text=prompt, image_url=f"data:image/jpeg;base64,{base64_image}"
        )
        response, _ = vl_adapter.generate_response(input_data, max_tokens=1024)
    except Exception as e:
        print(f"\nError during API call for image {os.path.basename(image_path)}: {e}")
        return []

    # 4. 解析响应并恢复坐标
    try:
        json_str = response[response.find("[") : response.rfind("]") + 1]
        if not json_str and "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        
        bbox_list = json_repair.loads(json_str)

        valid_bboxes = []
        scale_w, scale_h = resize_scale

        for item in bbox_list:
            bbox = item.get("bbox_2d", [])
            if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                # 接收模型在缩放后图像上的坐标
                x1, y1, x2, y2 = bbox
                
                # --- ↓↓↓ 新增：将坐标转换回原始图像尺寸 ↓↓↓ ---
                orig_x1 = int(x1 * scale_w)
                orig_y1 = int(y1 * scale_h)
                orig_x2 = int(x2 * scale_w)
                orig_y2 = int(y2 * scale_h)
                
                # 确保坐标在原始图像范围内
                orig_w, orig_h = original_size
                orig_x1, orig_y1 = max(0, orig_x1), max(0, orig_y1)
                orig_x2, orig_y2 = min(orig_w - 1, orig_x2), min(orig_h - 1, orig_y2)
                
                valid_bboxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
        
        if not valid_bboxes:
            print(f"Warning: No valid bboxes found for {os.path.basename(image_path)}. Raw response: {response[:200]}...")

        return valid_bboxes
    except Exception as e:
        print(f"\nFailed to parse response for {os.path.basename(image_path)}: {e}")
        print(f"Raw response was: {response}")
        return []
# --- 新的批量处理主函数 ---
def batch_process_directory(input_dir: str, output_dir: str, instruction: str):
    """
    批量处理一个目录中的所有图片。

    参数:
    - input_dir: 包含输入图片的文件夹路径。
    - output_dir: 保存可视化结果的文件夹路径。
    - instruction: 应用于所有图片的自然语言指令。
    """
    print("--- Starting Batch Processing ---")
    
    # 1. 检查路径和准备文件列表
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"No images found in '{input_dir}'")
        return

    print(f"Found {len(image_files)} images to process.")

    # 2. 初始化一次 VLM 适配器 (重要优化!)
    print("Initializing OpenAI Adapter...")
    try:
        vl_adapter = OpenAIAdapter(
            # api_key="EMPTY",
            # base_url="http://10.134.159.154:8000/v1",
            # model_name=None
            api_key = "sk-d12214631b904e87882434229779269a", 
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
            model_name = "qwen-vl-plus"
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI Adapter: {e}")
        return
        
    # 3. 循环处理每张图片
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        
        # 调用核心函数获取边界框
        bboxes = get_bboxes_for_image(vl_adapter, input_path, instruction)
        
        # 如果成功获取到边界框，则进行可视化并保存
        if bboxes:
            frame = cv2.imread(input_path)
            
            # 构建输出文件名
            base_name, _ = os.path.splitext(filename)
            output_filename = f"{base_name}_output.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 可视化
            visualize(
                frame,
                bboxes=bboxes,
                mask=None,
                save_path=output_path
            )
    
    print("\n--- Batch processing finished! ---")
    print(f"Results saved to '{output_dir}'")

def count_workers_in_image(vl_adapter, image_path: str, max_image_size: int = 768) -> int:
    """
    使用 VLM 检测并统计单张图片中的施工人员数量。

    参数:
    - vl_adapter: 已初始化的 OpenAIAdapter 实例。
    - image_path: 输入图片的路径。
    - max_image_size: 图像最长边的最大像素值。

    返回:
    - 检测到的施工人员数量 (int)，如果失败或未检测到则返回 0。
    """
    # 1. 加载和缩放图像 (与 get_bboxes_for_image 逻辑相同)
    original_frame = cv2.imread(image_path)
    if original_frame is None:
        print(f"Warning: Could not read image from path: {image_path}")
        return 0

    h, w = original_frame.shape[:2]
    if max(h, w) > max_image_size:
        scale = max_image_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_for_model = cv2.resize(original_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        frame_for_model = original_frame

    rgb_frame = frame_for_model[:, :, ::-1].copy()
    base64_image = _image_to_base64(rgb_frame)

    # 2. 构建专门用于计数的提示词 (这是关键区别)
    # prompt = (
    #     "Your task is to act as a person counter. Carefully examine the image and count every construction worker visible. "
    #     "A construction worker is typically identified by wearing a safety helmet and a work vest. "
    #     "Return your answer as a single integer number. Do not provide any other text, explanation, or sentences. "
    #     "For example, if you see 3 workers, your response should be just '3'."
    # )
    prompt = ("Your task is to act as a robotic arm counter. Carefully examine the image and count every robotic arm visible. ""Return your answer as a single integer number. Do not provide any other text, explanation, or sentences. ""For example, if you see 3 robotic arms, your response should be just '3'.")
    
    # 3. 调用 VLM
    try:
        input_data = vl_adapter.prepare_input(
            text=prompt, image_url=f"data:image/jpeg;base64,{base64_image}"
        )
        # 减少 max_tokens 因为我们只需要一个数字
        response, _ = vl_adapter.generate_response(input_data, max_tokens=32) 
    except Exception as e:
        print(f"\nError during API call for image {os.path.basename(image_path)}: {e}")
        return 0

    # 4. 解析响应以提取数字 (这是关键区别)
    try:
        # 使用正则表达式查找响应中的第一个数字
        match = re.search(r'\d+', response)
        if match:
            # 如果找到，则转换为整数并返回
            count = int(match.group(0))
            return count
        else:
            # 如果响应中没有数字，则认为未检测到
            print(f"Warning: No number found in response for {os.path.basename(image_path)}. Raw response: '{response}'")
            return 0
    except Exception as e:
        print(f"\nFailed to parse number from response for {os.path.basename(image_path)}: {e}")
        print(f"Raw response was: {response}")
        return 0

def batch_count_workers(input_dir: str, output_file: str):
    """
    批量处理目录中的所有图片，统计施工人员数量，并将结果保存到JSON文件。

    参数:
    - input_dir: 包含输入图片的文件夹路径。
    - output_file: 保存结果的JSON文件路径。
    """
    print("--- Starting Batch Processing for Worker Counting ---")
    
    # 1. 检查路径和准备文件列表
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    # 确保输出文件的目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"No images found in '{input_dir}'")
        return

    print(f"Found {len(image_files)} images to process.")

    # 2. 初始化 VLM 适配器 (与之前相同)
    print("Initializing OpenAI Adapter...")
    try:
        vl_adapter = OpenAIAdapter(
            api_key = "sk-d12214631b904e87882434229779269a", 
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
            model_name = "qwen-vl-plus"
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI Adapter: {e}")
        return
        
    # 3. 循环处理每张图片并存储结果
    results = {}
    for filename in tqdm(image_files, desc="Counting workers"):
        input_path = os.path.join(input_dir, filename)
        
        # 调用新的计数函数
        worker_count = count_workers_in_image(vl_adapter, input_path)
        
        # 将结果存入字典
        results[filename] = worker_count
        print(f"  -> {filename}: Found {worker_count} worker(s).")
    
    # 4. 将结果保存到JSON文件
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n--- Worker counting finished! ---")
        print(f"Results saved to '{output_file}'")
    except Exception as e:
        print(f"\nError saving results to JSON file: {e}")
        
# 使用示例
if __name__ == "__main__":
    # setup_all()

    # for centers in main_rs_iter(
    #     save_video=False,
    #     disable_vlm=False,
    #     video_source="camera",
    #     video_path="/data/shiqi/ImagePipelien/imagepipeline/output.mp4",
    # ):
    #     print(centers)

    #import os

    #os.environ["DISPLAY"] = ":0"  # 设置显示环境变量
    #main_qwen_demo()

    # shutdown_all()
    
    # 1. 设置你的输入图片文件夹
    INPUT_DIR = "./input_images_worker"  # <--- 修改这里
    
    # 2. 设置你的输出结果文件夹
    OUTPUT_DIR = "./test_images_worker" # <--- 修改这里
    
    # 3. 设置你想要检测的目标指令
    INSTRUCTION = "workers without seat belts" # <--- 修改这里
    # --- 配置结束 ---

    # 运行批量处理
    batch_process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        instruction=INSTRUCTION
    )
    # print("\n--- RUNNING WORKER COUNTING ---")
    # # 1. 设置你的输入图片文件夹
    # INPUT_DIR_COUNT = "./input_images_machine"  # <--- 修改这里
    
    # # 2. 设置你的输出结果文件名 (JSON格式)
    # OUTPUT_FILE_COUNT = "./machine_counts.json" # <--- 修改这里
    
    # # 运行新的批量计数处理
    # batch_count_workers(
    #     input_dir=INPUT_DIR_COUNT,
    #     output_file=OUTPUT_FILE_COUNT
    # )
