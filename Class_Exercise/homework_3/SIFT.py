import cv2
import numpy as np

img_full = cv2.imread('test_sift.png')

height, width, _ = img_full.shape
split_point = width // 2

# img1 是左半部分, img2 是右半部分
img1 = img_full[:, :split_point]
img2 = img_full[:, split_point:]

# 转换为灰度图
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
print("SIFT 检测器创建成功。")

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"图像1: 检测到 {len(kp1)} 个关键点。")
print(f"图像2: 检测到 {len(kp2)} 个关键点。")

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

print(f"找到了 {len(matches)} 组原始匹配。")

good_matches = []
ratio_thresh = 0.7

for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append([m])

print(f"筛选后剩余 {len(good_matches)} 个良好匹配。")

result_img = cv2.drawMatchesKnn(
    img1, kp1,         # 图像1
    img2, kp2,         # 图像2
    good_matches,      # 筛选后的匹配
    None,              # 输出图像
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

output_filename = 'sift_matches_result.jpg'
cv2.imshow('SIFT Matches', result_img)
cv2.imwrite(output_filename, result_img)
print(f"匹配结果已保存为 '{output_filename}'")
cv2.waitKey(0)