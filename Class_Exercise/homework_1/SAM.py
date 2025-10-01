import os
import torch
import cv2
import requests
from tqdm import tqdm
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def download_file(url, destination):
    """
    使用 requests 和 tqdm 下载文件并显示进度条。
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 确保请求成功

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 * 8 # 8KB

        progress_bar = tqdm(
            total=total_size_in_bytes, 
            unit='iB', 
            unit_scale=True,
            desc=f"正在下载 {os.path.basename(destination)}"
        )
        
        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("错误：下载的文件大小与预期不符。")
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        # 如果下载失败，删除不完整的文件
        if os.path.exists(destination):
            os.remove(destination)
        raise

# --------------------------------------------------------------------------
#                           --- 主程序开始 ---
# --------------------------------------------------------------------------

# -----------------
# 1. 模型设置与自动下载
# -----------------
# 定义不同SAM模型的信息 (文件名和下载链接)
SAM_MODELS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth"
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth"
    }
}

# <<< 在这里选择你想使用的模型版本 ('vit_b', 'vit_l', 'vit_h') >>>
SAM_MODEL_TYPE = "vit_b"
# -----------------------------------------------------------------

# 根据选择的模型类型，设置对应的权重文件路径和下载链接
SAM_CHECKPOINT_PATH = SAM_MODELS[SAM_MODEL_TYPE]["filename"]
SAM_URL = SAM_MODELS[SAM_MODEL_TYPE]["url"]

# 检查模型权重文件是否存在，如果不存在则自动下载
if not os.path.exists(SAM_CHECKPOINT_PATH):
    print(f"模型权重文件 '{SAM_CHECKPOINT_PATH}' 不存在。")
    download_file(SAM_URL, SAM_CHECKPOINT_PATH)
else:
    print(f"找到已存在的模型权重文件: '{SAM_CHECKPOINT_PATH}'")


# -----------------
# 2. 加载模型并创建自动掩码生成器
# -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用 {device} 设备加载模型...")

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device)
mask_generator = SamAutomaticMaskGenerator(model=sam) # 使用默认参数

# -----------------
# 3. 加载图像并执行分割
# -----------------
# <<< 将 'your_image.jpg' 替换为你的图片文件名 >>>
IMAGE_PATH = 'test.jpg'
# -----------------------------------------------

if not os.path.exists(IMAGE_PATH):
     print(f"错误：找不到图片文件 '{IMAGE_PATH}'，请检查文件名和路径。")
else:
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("开始自动分割整张图片，请稍候...")
    masks = mask_generator.generate(image_rgb)
    print(f"分割完成！共找到 {len(masks)} 个物体/区域。")

    # -----------------
    # 4. 可视化结果
    # -----------------
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=masks)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    sv.plot_image(annotated_image, (16, 16))
    cv2.imwrite('segmented_image_SAM.jpg', annotated_image)