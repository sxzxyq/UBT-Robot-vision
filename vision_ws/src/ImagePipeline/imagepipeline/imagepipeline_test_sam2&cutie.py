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
from utils import visualize_for_video, visualize_centers_for_video
from model_adapters import QwenVLAdapter
from tqdm import tqdm

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Realsense library not available. Skipping RealsenseCamera functionality.")


class ImagePipeline:
    def __init__(
        self, sam_checkpoint: str, sam_model_config: str, cutie_model, vl_adapter=None
    ):
        """
        多目标分割与跟踪管道（集成视觉语言模型）

        参数:
        sam_checkpoint: SAM模型权重路径
        sam_model_config: SAM模型类型配置 (i.e. "configs/sam2.1/sam2.1_hiera_l.yaml")
        cutie_model: 预加载的CUTIE模型实例
        vl_adapter: 视觉语言模型适配器实例（QwenVLAdapter）
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
        if self.vl_adapter is None:
            raise RuntimeError(
                "Visual Language Model adapter is not initialized. "
                "Cannot use get_bbox_from_vl or initialize_with_instruction."
            )
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

# ... (在你的所有类和函数定义之后) ...

def main_demo_no_gui():
    """
    一个用于在无GUI环境（如远程开发板）下测试 SAM2 和 CUTIE 的函数。
    它加载一张图片，使用预先定义的边界框，并将结果保存为文件。
    """
    print("Starting NO-GUI test for SAM2 and CUTIE...")

    # --- 1. 配置模型和图片路径 ---
    # 确保这些路径在你的开发板上是正确的
    sam_checkpoint = "./../sam2/checkpoints/sam2.1_hiera_large.pt"
    sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # 图片位置：确保这张图片也已经上传到了你的开发板上
    # 建议放在一个 'test_images' 文件夹里
    image_path = "./imagepipeline/test_images/image_test.jpg" # <--- 修改为你的图片路径
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"无法加载图片，请检查路径: {image_path}")
    
    rgb_frame = frame[:, :, ::-1].copy()

    # --- 2. 初始化模型 (不带VLM) ---
    print("Initializing models...")
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

        pipeline = ImagePipeline(
            sam_checkpoint, sam_model_config, cutie_model, vl_adapter=None
        )
        print("ImagePipeline initialized without VLM.")

    # --- 3. 定义目标边界框 (Bounding Box) ---
    # !!! 关键步骤 !!!
    # 在这里粘贴你从本地电脑上运行 get_coordinates.py 后得到的坐标
    manual_bboxes = [[362, 77, 427, 143]] # <--- PASTE YOUR COORDINATES HERE

    print(f"Using predefined bounding boxes: {manual_bboxes}")

    # --- 4. 使用边界框初始化跟踪 ---
    print("Initializing tracker with predefined bounding boxes...")
    combined_mask = pipeline.initialize_masks(
        frame=rgb_frame,
        bboxes=manual_bboxes,
    )

    # 可视化初始分割结果并保存
    print("Initial segmentation generated. Saving to 'initial_segmentation.png'...")
    vis_frame_initial = frame.copy()
    # 你的 visualize 函数已经支持 save_path，这很好！
    visualize(vis_frame_initial, bboxes=manual_bboxes, mask=combined_mask, save_path="initial_segmentation.png")
    centers_initial = compute_mask_centers(combined_mask, "centroid")
    # visualize_centers 可能需要修改以支持保存，如果它目前只显示图像的话
    # 假设 visualize_centers 绘制后返回图像，或者我们可以直接在图像上绘制
    visualize_centers(vis_frame_initial, centers=centers_initial) # 这会修改 vis_frame_initial
    cv2.imwrite("initial_segmentation_with_centers.png", vis_frame_initial)


    # --- 5. 模拟视频流并进行跟踪 ---
    print("\n--- Simulating tracking and saving final result... ---")
    # 模拟处理10帧，然后保存最后一帧的结果
    updated_mask = combined_mask
    for i in range(10):
        # 在真实场景中，这里会是 new_frame = camera.get_frame()
        # 我们用同一张图片来模拟
        new_frame_rgb = rgb_frame.copy()

        updated_mask, _ = pipeline.update_masks(new_frame_rgb)
        print(f"Simulated frame {i+1}/10 processed.")

    # 计算最后一帧的中心点并保存结果
    centers = compute_mask_centers(updated_mask, "centroid")
    print(f"Final centers after 10 frames: {centers}")
    
    vis_frame_tracked = frame.copy()
    visualize(vis_frame_tracked, mask=updated_mask, save_path="tracked_output.png")
    visualize_centers(vis_frame_tracked, centers=centers) # 修改 vis_frame_tracked
    cv2.imwrite("tracked_output_with_centers.png", vis_frame_tracked)

    print("\nTest finished. Check the saved image files:")
    print("- initial_segmentation.png")
    print("- initial_segmentation_with_centers.png")
    print("- tracked_output.png")
    print("- tracked_output_with_centers.png")

def main_demo_video_1(video_path: str, output_path: str, manual_bboxes: list):
    """
    一个用于在无GUI环境下测试视频跟踪的函数。
    它加载一个视频文件，使用第一帧的预定义边界框进行初始化，
    然后逐帧跟踪并将结果保存到一个新的视频文件中。

    参数:
    video_path: 输入视频文件的路径。
    output_path: 保存结果视频的路径。
    manual_bboxes: 用于初始化第一帧的边界框列表 [[x1, y1, x2, y2], ...]。
    """
    print("Starting video processing test for SAM2 and CUTIE...")

    # --- 1. 配置模型路径 ---
    sam_checkpoint = "./../sam2/checkpoints/sam2.1_hiera_large.pt"
    sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # --- 2. 初始化模型 (不带VLM) ---
    print("Initializing models...")
    pipeline = None
    with torch.no_grad():
        try:
            cutie_model = get_default_model()
        except ValueError as e:
            if "GlobalHydra is already initialized" in str(e):
                GlobalHydra.instance().clear()
                cutie_model = get_default_model()
            else:
                raise
        print("CUTIE model loaded successfully.")

        pipeline = ImagePipeline(
            sam_checkpoint, sam_model_config, cutie_model, vl_adapter=None
        )
        print("ImagePipeline initialized without VLM.")

    # --- 3. 设置视频输入和输出 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件，请检查路径: {video_path}")

    # 获取视频属性用于输出
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Video opened successfully: {video_path}")
    print(f"Output will be saved to: {output_path} with {fps} FPS.")

    frame_count = 0
    try:
        # --- 4. 读取第一帧并进行初始化 ---
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("无法读取视频的第一帧。")

        print(f"Initializing tracker with predefined bounding boxes on the first frame: {manual_bboxes}")
        rgb_frame = frame[:, :, ::-1].copy()
        
        # 使用边界框初始化跟踪
        combined_mask = pipeline.initialize_masks(
            frame=rgb_frame,
            bboxes=manual_bboxes,
        )

        # 可视化第一帧的结果并写入输出视频
        vis_frame = frame.copy()
        visualize(vis_frame, bboxes=manual_bboxes, mask=combined_mask)
        centers = compute_mask_centers(combined_mask, "centroid")
        visualize_centers(vis_frame, centers=centers)
        video_writer.write(vis_frame)
        frame_count += 1
        print(f"Frame {frame_count}: Initialization complete.")


        # --- 5. 循环处理视频的剩余帧 ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            rgb_frame = frame[:, :, ::-1].copy()
            
            # 更新掩码进行跟踪
            updated_mask, _ = pipeline.update_masks(rgb_frame)
            
            # 可视化当前帧的跟踪结果
            vis_frame = frame.copy()
            visualize(vis_frame, mask=updated_mask,blend_alpha=1.0)
            centers = compute_mask_centers(updated_mask, "centroid")
            visualize_centers(vis_frame, centers=centers)
            
            # 将处理后的帧写入输出视频
            video_writer.write(vis_frame)
            frame_count += 1
            if frame_count % 30 == 0: # 每30帧打印一次进度
                print(f"Processed {frame_count} frames...")

    except Exception as e:
        print(f"An error occurred during video processing: {e}")
    finally:
        # --- 6. 释放资源 ---
        print("Releasing video resources...")
        cap.release()
        video_writer.release()
        print(f"Processing finished. Output video saved to: {output_path}")


def main_demo_video(video_path: str, output_path: str, manual_bboxes: list):
    """
    [调试版] 一个用于在无GUI环境下处理视频的函数。
    增加了详细的打印输出来诊断视频时长不匹配的问题。
    """
    print("Starting background video processing in DEBUG mode...")

    # --- 1. 初始化模型 ---
    # ... (这部分代码保持不变) ...
    sam_checkpoint = "./../sam2/checkpoints/sam2.1_hiera_large.pt"
    sam_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    pipeline = None
    with torch.no_grad():
        try:
            cutie_model = get_default_model()
        except ValueError as e:
            if "GlobalHydra is already initialized" in str(e):
                GlobalHydra.instance().clear()
                cutie_model = get_default_model()
            else:
                raise
        pipeline = ImagePipeline(sam_checkpoint, sam_model_config, cutie_model, vl_adapter=None)
    print("Models initialized successfully.")

    # --- 2. 设置视频输入和输出 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    # --- DEBUG: 打印所有重要的视频属性 ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # 使用浮点数以获得更精确的值
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("\n" + "="*30)
    print("      VIDEO METADATA DIAGNOSTICS      ")
    print("="*30)
    print(f"Input Video Path: '{video_path}'")
    print(f"Reported FPS: {fps}")
    print(f"Reported Total Frames: {total_frames}")
    if fps > 0:
        print(f"Calculated Duration: {total_frames / fps:.2f} seconds")
    print("="*30 + "\n")
    # --- END DEBUG ---

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 使用读取到的 fps 来创建写入器
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    processed_frames_count = 0
    try:
        # --- 3. 处理第一帧 (初始化) ---
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read the first frame of the video.")
        processed_frames_count += 1
        
        # ... (初始化逻辑保持不变) ...
        rgb_frame = frame[:, :, ::-1].copy()
        combined_mask = pipeline.initialize_masks(frame=rgb_frame, bboxes=manual_bboxes)
        vis_frame = frame.copy()
        #visualize(vis_frame, bboxes=manual_bboxes, mask=combined_mask)
        #centers = compute_mask_centers(combined_mask, "centroid")
        #visualize_centers(vis_frame, centers=centers)
        vis_frame = visualize_for_video(frame, combined_mask, manual_bboxes)
        centers = compute_mask_centers(combined_mask, "centroid")
        vis_frame = visualize_centers_for_video(vis_frame, centers)
        video_writer.write(vis_frame)

        # --- 4. 循环处理剩余帧 (跟踪) ---
        for _ in tqdm(range(1, total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                print(f"\n[WARNING] cap.read() returned False. Loop stopped earlier than expected.")
                break

            processed_frames_count += 1
            # ... (跟踪逻辑保持不变) ...
            rgb_frame = frame[:, :, ::-1].copy()
            updated_mask, _ = pipeline.update_masks(rgb_frame)
            vis_frame = frame.copy()
            #visualize(vis_frame, mask=updated_mask)
            #centers = compute_mask_centers(updated_mask, "centroid")
            #visualize_centers(vis_frame, centers=centers)
            vis_frame = visualize_for_video(frame, updated_mask, None, blend_alpha=1.0)
            centers = compute_mask_centers(updated_mask, "centroid")
            vis_frame = visualize_centers_for_video(vis_frame, centers)
            video_writer.write(vis_frame)

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
    finally:
        # --- DEBUG: 打印最终的处理结果统计 ---
        print("\n" + "="*30)
        print("        PROCESSING SUMMARY        ")
        print("="*30)
        print(f"Total frames actually processed: {processed_frames_count}")
        print(f"Expected to process: {total_frames}")
        if fps > 0:
            output_duration = processed_frames_count / fps
            print(f"FPS used for writing: {fps}")
            print(f"Expected output duration: {output_duration:.2f} seconds")
        print("="*30 + "\n")
        # --- END DEBUG ---

        cap.release()
        video_writer.release()
        print(f"Processing finished. Final video saved to '{output_path}'")
        
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
    #main_demo()

    # shutdown_all()
    
    #main_demo_no_gui()
    input_video = "./imagepipeline/test_images/video_test.mp4" 
    output_video = "./imagepipeline/result_videos/tracked_video_output.mp4"
    first_frame_bboxes = [[704, 151, 825, 250]]
    main_demo_video(
        video_path=input_video,
        output_path=output_video,
        manual_bboxes=first_frame_bboxes
    )
