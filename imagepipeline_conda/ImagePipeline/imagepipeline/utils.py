from typing import Tuple,List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import cv2

# os.environ['DISPLAY'] = "109.105.4.86:1.0"

def add_mask(
    mask: np.ndarray, 
    bgr_image: np.ndarray, 
    blend_alpha: float = 0.5
) -> np.ndarray:
    """
    在 BGR 图像上添加彩色掩码。

    参数:
    mask: 多目标mask数组，每个像素值为对象ID (0表示背景)。
    bgr_image: OpenCV 读取的原始 BGR 格式图像。
    blend_alpha: 混合的透明度。如果设置为 1.0，则完全用掩码颜色替换。
                 如果设置为 0.5，则是半透明覆盖。
    返回:
    处理后的 BGR 格式图像。
    """
    # 创建一个与输入图像相同大小的副本用于绘制
    output_image = bgr_image.copy()

    # 获取所有唯一的对象ID（不包括背景0）
    unique_ids = np.unique(mask)
    
    for obj_id in unique_ids:
        if obj_id == 0:
            continue

        # 找到当前对象的掩码区域
        object_mask_area = (mask == obj_id)
        
        # 获取颜色 (matplotlib的tab10返回的是[0,1]范围的RGB)
        #color_rgb = plt.cm.tab10((obj_id % 10) / 10)[:3]
        
        # 将颜色转换为OpenCV能用的[0,255]范围的BGR
        #color_bgr = tuple(int(c * 255) for c in color_rgb[::-1]) # RGB -> BGR
        color_bgr = (0, 0, 0)

        # --- 核心逻辑：混合或替换 ---
        if blend_alpha >= 1.0:
            # 如果alpha为1或更高，则直接用纯色替换
            output_image[object_mask_area] = color_bgr
        else:
            # 否则，进行Alpha混合
            # 提取掩码区域的原始图像部分
            roi = output_image[object_mask_area]
            
            # 创建一个与ROI相同大小的纯色块
            color_block = np.full(roi.shape, color_bgr, dtype=np.uint8)
            
            # 使用cv2.addWeighted进行混合
            blended_roi = cv2.addWeighted(color_block, blend_alpha, roi, 1 - blend_alpha, 0)
            
            # 将混合后的区域放回原图
            output_image[object_mask_area] = blended_roi
            
    return output_image

def visualize(
    image: np.ndarray,
    bboxes: Optional[List[List[int]]] = None,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dpi: int = 100,
    return_result: bool = False,
) -> None:
    """
    可视化图像、边界框和mask

    参数:
    image: RGB格式的输入图像 [H, W, 3]
    bboxes: 边界框列表 [[x1, y1, x2, y2], ...]
    mask: 多目标mask数组，每个像素值为对象ID (0表示背景)
    save_path: 图片保存路径 (None则显示)
    dpi: 输出图像分辨率
    """
    # 创建画布
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Visualization Results", fontsize=16)
    axes = axs.flatten()

    # 子图1: 原始图像
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 子图2: 带边界框的图像
    axes[1].imshow(image)
    if bboxes:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            axes[1].add_patch(rect)
    axes[1].set_title("With Bounding Boxes")
    axes[1].axis("off")

    # 子图3: 单独mask
    if mask is not None:
        # 生成彩色mask (忽略0背景)
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        unique_ids = np.unique(mask)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            color = plt.cm.get_cmap("tab10")(obj_id % 10)[
                :3
            ]  # 使用tab10颜色循环
            colored_mask[mask == obj_id] = np.array(color) * 255
        axes[2].imshow(colored_mask)
        axes[2].set_title("Segmentation Mask")
    else:
        axes[2].imshow(np.zeros_like(image))
        axes[2].set_title("No Mask Available")
    axes[2].axis("off")

    # 子图4: 叠加mask的图像
    axes[3].imshow(image)
    if mask is not None:
        # 创建半透明覆盖层
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            color = plt.cm.get_cmap("tab10")(obj_id % 10)
            overlay[mask == obj_id] = [*color[:3], 0.5]  # RGBA格式

        axes[3].imshow(overlay)
    axes[3].set_title("Image with Mask Overlay")
    axes[3].axis("off")

    # 调整布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()
    if return_result:
        return overlay

def compute_mask_centers(
    mask: np.ndarray, method: str = "centroid"
) -> dict:
    """
    计算多目标mask中每个对象的中心点

    参数:
    mask: 多目标mask数组，每个像素值为对象ID (0表示背景)
    method: 中心计算方法，可选'centroid'（质心）或'bounding_box'（边界框中心）

    返回:
    中心点坐标列表 [(x1, y1), (x2, y2), ...]，按对象ID升序排列
    """
    centers = {}
    unique_ids = np.unique(mask)

    for obj_id in unique_ids:
        if obj_id == 0:
            continue  # 跳过背景

        if method == "centroid":
            # 计算质心：所有像素坐标的平均值
            y_indices, x_indices = np.where(mask == obj_id)
            cx = np.mean(x_indices)
            cy = np.mean(y_indices)

        elif method == "bounding_box":
            # 计算边界框中心：包围盒的几何中心
            obj_region = mask == obj_id
            rows = np.any(obj_region, axis=1)
            cols = np.any(obj_region, axis=0)

            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]

            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0

        else:
            raise ValueError(
                f"Unsupported method: {method}. Choose 'centroid' or 'bounding_box'"
            )
        
        centers[obj_id] = (cx, cy) # 按ID存入字典

    return centers

def visualize_centers(
    image: np.ndarray,
    centers: List[Tuple[float, float]],
    save_path: Optional[str] = None,
    dpi: int = 100
) -> None:
    """
    可视化图像及中心点
    
    参数:
    image: RGB格式的输入图像 [H, W, 3]
    centers: 中心点坐标列表 [(x1, y1), (x2, y2), ...]
    save_path: 图片保存路径 (None则显示)
    dpi: 输出图像分辨率
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title("Image with Object Centers")
    
    # 绘制中心点（使用不同颜色区分对象）
    for idx, (cx, cy) in enumerate(centers):
        color = plt.cm.tab10(idx % 10)  # 使用tab10颜色循环
        plt.scatter(cx, cy, s=120, edgecolors='black', 
                   linewidths=1.5, facecolor=color,
                   marker='o', label=f'Object {idx+1}')
    
    # 添加图例（显示在图像右侧）
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.axis('off')
    
    # 调整布局并保存/显示
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Center visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
def visualize_for_video(
    frame: np.ndarray,
    mask: np.ndarray,
    bboxes: Optional[List[List[int]]] = None,
    blend_alpha: float = 0.5,
) -> np.ndarray:
    """
    (非阻塞) 在单个帧上可视化掩码和可选的边界框，并返回结果帧。
    此函数使用OpenCV，速度快，适合视频流。
    """
    # 1. 使用你现有的 add_mask 函数来创建带掩码的背景
    # add_mask 返回的是 BGR 格式，可以直接用于 OpenCV
    vis_frame = add_mask(mask, frame, blend_alpha)

    # 2. 如果有边界框，使用 OpenCV 绘制它们
    if bboxes is not None:
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(p) for p in bbox]
            # 在 vis_frame 上用红色绘制矩形
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # BGR
            
    return vis_frame


def visualize_centers_for_video(
    frame: np.ndarray,
    centers: dict, # 注意：我们将使用字典以获得更好的可读性
) -> np.ndarray:
    """
    (非阻塞) 在单个帧上可视化中心点，并返回结果帧。
    """
    # 直接在传入的帧上操作
    for obj_id, center in centers.items():
        cx, cy = int(center[0]), int(center[1])
        
        # 使用 tab10 调色板的颜色 (BGR格式)
        color_rgb = plt.cm.tab10((obj_id % 10) / 10)[:3]
        color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])

        # 绘制一个带边框的圆点
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1) # 白色背景
        cv2.circle(frame, (cx, cy), 6, color_bgr, -1)      # 彩色前景
        
        # 绘制对象ID文本
        cv2.putText(frame, str(obj_id), (cx + 10, cy + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3) # 白色描边
        cv2.putText(frame, str(obj_id), (cx + 10, cy + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    
    return frame

# 使用示例
if __name__ == "__main__":
    # 测试数据
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_bboxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[150:250, 150:250] = 1
    test_mask[350:450, 350:450] = 2
    
    # 调用可视化函数
    visualize(test_image, test_bboxes, test_mask, save_path="demo_visualization.png")

    # 创建测试mask
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[150:250, 150:250] = 1  # 对象1：100x100正方形
    test_mask[350:450, 350:450] = 2  # 对象2：100x100正方形
    
    # 计算质心中心
    centroid_centers = compute_mask_centers(test_mask, method='centroid')
    print("Centroid Centers:", centroid_centers)  # 应输出 [(199.5, 199.5), (399.5, 399.5)]
    
    # 计算包围盒中心
    bbox_centers = compute_mask_centers(test_mask, method='bounding_box')
    print("BBox Centers:", bbox_centers)          # 应输出 [(199.5, 199.5), (399.5, 399.5)]