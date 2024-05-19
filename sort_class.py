"""按照类别3 1 0 2的顺序对bbox重新排序，txt每行的第一个值代表类别
对一整个文件夹进行这个操作并写入原始位置
"""

import os

# 指定文件夹路径
folder_path = 'bbox'

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    # 检查文件是否为文本文件
    if filename.endswith('.txt'):
        # 构建文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 读取文本文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # 将每行内容拆分为索引和数值
        data = [line.split() for line in lines]
        
        # 按照给定的索引顺序对数据进行排序
        sorted_data = sorted(data, key=lambda x: [3, 1, 0, 2].index(int(x[0])))
        
        # 将排序后的数据重新组合成字符串
        sorted_lines = [' '.join(line) for line in sorted_data]
        
        # 将排序后的结果写入原文件位置
        with open(file_path, 'w') as file:
            file.write('\n'.join(sorted_lines))