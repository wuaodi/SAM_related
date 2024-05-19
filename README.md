# SAM_related
利用SAM开发的一些相关代码

1、SAM_grid_test.py 为anything模式

2、SAM_grid_dir.py 对整个文件夹图像进行SAM分割，结果按照区域大小，从大到小赋语义，颜色随机
python SAM_grid_dir.py --image_dir images/sub1 --output_dir mask/sub1

3、SAM_bbox_test.py 输入单张图像和bbox prompt出分割结果

4、SAM_bbox_dir.py 基于yolo检测结果，按照类别来赋予颜色，相同的类别赋予mask相同的颜色，不同的类别赋予不同的颜色

首先使用sort_class.py代码调整yolo检测结果中bbox类别的顺序，以便后续按照这个顺序给mask赋予分割结果，例如sensor和cabin这两个类别，
为了避免先给sensor赋予mask后被cabin挡住，则应该首先给cabin赋予mask再给sensor赋予mask

5、SAM_point_test.py 输入point prompt，输出这些点合成一个mask结果

6、SAM_point_dir.py 从语义标签中随机采样一些点来做SAM, 语义标签和原始图像相同名字，各放一个文件夹

7、IoU.py 不同类别用不同颜色表示，计算各个类别的IoU

8、Circle.py 根据SAM结果提取圆形边框

9、slip_aug.py 使用SAM和CLIP对原图进行增强, 相同语义的区域赋予相同的颜色
