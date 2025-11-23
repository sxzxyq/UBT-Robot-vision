# 文件: ~/Workspace/image_ws/src/image_inverter/image_inverter/helmet_detector_base.py

# ========================== 关键修正 - 开始 ==========================
# 确保模块搜索路径是正确的
import sys
import os

image_pipeline_path = os.path.abspath(os.path.expanduser('~/Workspace/imagepipeline_conda/ImagePipeline'))
if image_pipeline_path not in sys.path:
    sys.path.insert(0, image_pipeline_path)
# ========================== 关键修正 - 结束 ==========================

import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import json_repair

try:
    from imagepipeline.model_adapters import OpenAIAdapter
except ImportError as e:
    print(f"CRITICAL ERROR in helmet_detector_base.py: Failed to import OpenAIAdapter. Error: {e}")
    class OpenAIAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenAIAdapter could not be imported due to a path issue.")

class HelmetDetectorBase:
    """
    一个只封装了VLM安全帽检测逻辑的基础类。
    (这是从我们之前成功的 helmet_detector.py 中提取的)
    """
    # def __init__(self, model_api_base_url="http://10.134.159.154:8000/v1"):
    #     try:
    #         self.vl_adapter = OpenAIAdapter(api_key="EMPTY", base_url=model_api_base_url)
    #         print("OpenAI Adapter initialized successfully.")
    #     except Exception as e:
    #         print(f"FATAL: Failed to initialize OpenAI Adapter: {e}")
    #         self.vl_adapter = None
    
    # def __init__(self, api_key: str, base_url: str, model_name: str):
    #     """
    #     初始化安全帽检测器。

    #     :param api_key: 用于API调用的密钥。对于本地模型可以是 "EMPTY"，对于云服务则是真实密钥。
    #     :param base_url: VLM服务的根URL (例如 "http://10.134.159.154:8000/v1" 或 "https://dashscope.aliyuncs.com/compatible-mode/v1")。
    #     :param model_name: 要使用的具体模型名称 (例如 "local-vlm" 或 "qwen-vl-plus")。
    #     """
    #     dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    #     if not dashscope_api_key:
    #         print("警告: 未设置环境变量 DASHSCOPE_API_KEY，无法初始化通义千问检测器。")
    #     else:
    #         try:
    #             # 将接收到的参数传递给 OpenAIAdapter
    #             self.vl_adapter = OpenAIAdapter(
    #                 api_key=dashscope_api_key,
    #                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #                 model_name="qwen-vl-plus"  # 通义千问的视觉语言模型
    #             )
    #             print(f"✅ VLM Adapter initialized successfully for model '{self.model_name}' at '{self.base_url}'")
    #         except Exception as e:
    #             print(f"❌ FATAL: Failed to initialize VLM Adapter: {e}")
                
    def __init__(self, 
                 api_key: str = None, 
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                 model_name: str = "qwen-vl-plus"):
        """
        初始化安全帽检测器。
        参数带有默认值，优先使用通义千问，但仍然可以被覆盖。
        """
        # 如果 api_key 没有被直接提供，则尝试从环境变量中获取
        final_api_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY")
        
        if not final_api_key:
            raise ValueError("API key must be provided either as an argument or via DASHSCOPE_API_KEY environment variable.")
            
        self.api_key = final_api_key
        self.base_url = base_url
        self.model_name = model_name
        self.vl_adapter = None

        try:
            self.vl_adapter = OpenAIAdapter(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name
            )
            print(f"✅ VLM Adapter initialized successfully for model '{self.model_name}' at '{self.base_url}'")
        except Exception as e:
            print(f"❌ FATAL: Failed to initialize VLM Adapter: {e}")

    def _image_to_base64(self, image: np.ndarray) -> str:
        # VLM需要RGB格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def detect(self, frame: np.ndarray, instruction: str = "A human") -> list:
        if self.vl_adapter is None:
            print("Error: vl_adapter is not initialized. Cannot perform detection.")
            return []

        base64_image = self._image_to_base64(frame)
        prompt = (
            f"Your task is to act as a precise object detector.\n"
            f"In the image, find the exact location of ALL objects described as: '{instruction}'.\n"
            f"The bounding box must be as tight as possible around ONLY A human.\n"
            f"Return bboxes for ALL matching objects in this format:\n"
            '[{"bbox_2d": [x1,y1,x2,y2], "label": "A human"}, ...]'
            f"if no {instruction} existed in the image, just return 'NO'"
        )
        
        try:
            input_data = self.vl_adapter.prepare_input(text=prompt, image_url=f"data:image/jpeg;base64,{base64_image}")
            response, _ = self.vl_adapter.generate_response(input_data, max_tokens=1024)
        except Exception as e:
            print(f"Error during API call to the VL model: {e}")
            return []

        try:
            json_str = response[response.find("[") : response.rfind("]") + 1]
            if not json_str and "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            
            bbox_list = json_repair.loads(json_str)

            valid_bboxes = []
            h, w, _ = frame.shape
            for item in bbox_list:
                bbox = item.get("bbox_2d", [])
                if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
                    valid_bboxes.append([x1, y1, x2, y2])
            
            return valid_bboxes
        except Exception as e:
            print(f"Failed to parse VL model response: {e}\nRaw response: {response}")
            return []