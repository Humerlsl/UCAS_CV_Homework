import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt

# -----------------
# 1. 加载图像
# -----------------
image_path = 'test.jpg'
image = cv2.imread(image_path)

# 将图像从BGR颜色空间转换为RGB（matplotlib需要RGB格式）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------
# 2. 准备数据
# -----------------
# 将图像数据重塑为 (像素总数, 3) 的二维数组，其中3代表R,G,B三个通道
# 这是为了让scikit-learn的MeanShift能够处理它
pixel_list = image_rgb.reshape((-1, 3))

# -----------------
# 3. 应用均值移动算法
# -----------------
# estimate_bandwidth——带宽参数
# quantile——分位数
# n_samples——采样点数
bandwidth = estimate_bandwidth(pixel_list, quantile=0.1, n_samples=1500)

# 创建MeanShift模型实例
# bin_seeding=True 网格法加速
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# 训练模型（执行聚类）
ms.fit(pixel_list)

# 获取每个像素所属的聚类标签
labels = ms.labels_

# 获取所有聚类中心的颜色值
cluster_centers = ms.cluster_centers_

# 打印找到的聚类中心数量
print(f"找到的聚类中心数量: {len(np.unique(labels))}")

# -----------------
# 4. 重新构建图像
# -----------------
# 创建一个与原始图像大小相同的空白图像
segmented_image_data = np.zeros_like(pixel_list)

# 将每个像素的颜色设置为其所属聚类中心的颜色
for i in range(len(segmented_image_data)):
    segmented_image_data[i] = cluster_centers[labels[i]]

# 将一维的像素列表重新构建为原始图像的维度
segmented_image = segmented_image_data.reshape(image_rgb.shape)

# -----------------
# 5. 显示结果
# -----------------
plt.figure(figsize=(10, 5))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

# 显示分割后的图像
plt.subplot(1, 2, 2)
plt.title('Mean Shift Segmented Image')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()

# 保存分割后的图像
segmented_image_bgr = cv2.cvtColor(segmented_image.astype('uint8'), cv2.COLOR_RGB2BGR)
cv2.imwrite('segmented_image_mean_shift.jpg', segmented_image_bgr)