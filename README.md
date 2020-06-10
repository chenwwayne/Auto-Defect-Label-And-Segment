## 基于AttentionGAN和区域生长分割算法的缺陷标注与分割

### segImage.py中segImage函数：

输入：缺陷原图input、分割后的二值Mask

输出：画出标注框的图

流程：

- 对二值Mask求导，由于输入是二维矩阵，所以得到二个方向的导数矩阵cx, cy
- 两个导数矩阵取绝对值相加，其中导数不为0的像素即为二值Mask的边界像素，得到类型为bool的矩阵
- 以标注框为红色为例，取出input的红色通道, 与bool型矩阵相比取最大。由于bool型矩阵不是False 就是True,True即为最大值255，则对应的边界像素取最大即为红色框

label.py中segImage函数：

区域生长imreconstruct的意义在哪？