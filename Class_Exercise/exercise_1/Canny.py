# 导入所需的库
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# --- 1. 准备工作：加载图像 ---
try:
    print("正在从网络加载测试图像...")
    url = 'http://www.lenna.org/len_std.jpg'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # 将 PIL 图像转换为 OpenCV 格式 (BGR)
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # 将图像转换为灰度图，因为边缘检测通常在单通道图像上进行
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("图像加载成功！")
except Exception as e:
    print(f"图像加载失败，将创建一个备用测试图像。错误: {e}")
    # 如果网络加载失败，创建一个备用测试图像
    gray_image = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(gray_image, (50, 50), (250, 250), 255, -1)
    cv2.circle(gray_image, (150, 150), 80, 0, -1)


# --- 2. Canny 边缘检测 ---
# 调用 OpenCV 的 Canny 函数
# 参数说明:
#   - gray_image: 输入的灰度图像
#   - threshold1: 第一个阈值（低阈值）
#   - threshold2: 第二个阈值（高阈值）
# 梯度值高于 threshold2 的像素被认为是“强边缘”。
# 梯度值介于 threshold1 和 threshold2 之间的像素被认为是“弱边缘”。
# 梯度值低于 threshold1 的像素被抛弃。
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(gray_image, low_threshold, high_threshold)


# --- 3. 显示结果 ---
# 使用 matplotlib 来并排显示原始图和结果图
plt.figure(figsize=(10, 5))

# 显示原始灰度图
plt.subplot(1, 2, 1)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# 显示 Canny 边缘检测结果
plt.subplot(1, 2, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

# 展示图像
plt.tight_layout()
plt.show()