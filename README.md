## 基于AttentionGAN和区域生长分割算法的缺陷标注与分割

## Introduction

This repo based on [AttentionGAN](https://github.com/Ha0Tang/AttentionGAN)  result. Using AttentionGAN  to generate saliency image and target image (In this repo., as defect-free image). Using defect image, saliency image and defect-free image to realisze automatic  defect labeling  and segmentation of defects.

## sRequirements

- cv2
- numpy
- skimage

## AttentionGAN Result

<img src="https://github.com/ischansgithub/Auto-Defect-Label-And-Segment/blob/master/README/real.png" width="500">

## Detail of algorithm 

### 函数segImage：

输入：缺陷原图input、分割后的二值Mask

输出：画出标注框的图

流程：

- 对二值Mask求导，由于输入是二维矩阵，所以得到二个方向的导数矩阵cx, cy
- 两个导数矩阵取绝对值相加，其中导数不为0的像素即为二值Mask的边界像素，得到类型为bool的矩阵
- 以标注框为红色为例，取出input的红色通道, 与bool型矩阵相比取最大。由于bool型矩阵不是False 就是True,True即为最大值255，则对应的边界像素取最大即为红色框

### 形态学重建imreconstruct



## References