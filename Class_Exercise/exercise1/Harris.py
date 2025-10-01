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
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("图像加载成功！")
except Exception as e:
    print(f"图像加载失败，将创建一个备用测试图像。错误: {e}")
    # 如果网络加载失败，创建一个备用的棋盘格图像，它有清晰的角点
    gray_image = np.zeros((300, 300), dtype=np.uint8)
    square_size = 30
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 0:
                cv2.rectangle(gray_image, (i*square_size, j*square_size), 
                              ((i+1)*square_size, (j+1)*square_size), 255, -1)
    image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


# --- 2. Harris 角点检测 ---
# Harris 角点检测要求输入是 float32 类型的图像
gray_float = np.float32(gray_image)

# 调用 OpenCV 的 cornerHarris 函数
# 参数说明:
#   - gray_float: 输入的 float32 灰度图
#   - blockSize: 用于角点检测的邻域大小（窗口大小）
#   - ksize: Sobel 算子的孔径参数（用于计算梯度）
#   - k: Harris 检测器方程中的自由参数 k
dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

# --- 3. 标记和显示结果 ---
# 为了更好地可视化，我们将角点响应图进行扩张，使角点更明显
dst = cv2.dilate(dst, None)

# 创建一个副本图像用于绘制角点
image_with_corners = image.copy()

# 设置一个阈值来确定哪些是角点
# 我们将 dst 中大于 1% 最大值的点标记为角点
threshold = 0.01 * dst.max()
image_with_corners[dst > threshold] = [0, 0, 255] # 用红色标记角点

# --- 4. 显示结果 ---
# 使用 matplotlib 显示原始图和结果图
plt.figure(figsize=(10, 5))

# 显示原始彩色图
plt.subplot(1, 2, 1)
plt.title('Original Image')
# Matplotlib 显示的是 RGB，而 OpenCV 是 BGR，需要转换
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 显示 Harris 角点检测结果
plt.subplot(1, 2, 2)
plt.title('Harris Corners')
plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 展示图像
plt.tight_layout()
plt.show()