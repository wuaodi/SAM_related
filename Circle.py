'''根据SAM结果提取圆形边框'''

import cv2
import numpy as np

# 读取二值图像
binary_image = cv2.imread('mask_0.png', cv2.IMREAD_GRAYSCALE)

# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 在原图上绘制出检测到的圆
result = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像

# 遍历每个轮廓并拟合圆
for contour in contours:
    # 拟合最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(contour)
    print(x,y,radius)

    # 将圆心坐标和半径转为整数
    center = (int(x), int(y))
    radius = int(radius)

    # 在原图上绘制出检测到的圆
    cv2.circle(result, center, radius, (0, 255, 0), 2)

    # 在圆心位置绘制十字标记
    cv2.drawMarker(result, center, (255, 0, 0), cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # 输出圆心坐标和半径
    print(f'圆心坐标: {center}')
    print(f'圆半径: {radius}')

# 显示带有拟合圆和坐标轴的图像
cv2.imwrite('result.jpg', result)