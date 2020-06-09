# -*- coding:utf-8 -*-
import os
import os.path as osp
import cv2
import numpy as np
from math import ceil, floor
from skimage import measure
from segImage import segImage

def imreconstruct(marker, mask, SE=np.ones([3,3])):
    """
    描述：以mask为约束，连续膨胀marker，实现形态学重建，其中mask >= marker
    参数：
        - marker 标记图像，单通道/三通道图像
        - mask   模板图像，与marker同型
        - conn   联通性重建结构元，参照：matlab::imreconstruct::conn参数，默认为8联通
    """
    while True:
        marker_pre = marker
        # 膨胀
        dilation = cv2.dilate(marker, kernel=SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    return marker

def main():
    saliency_root = './images/saliency/'
    real_root = './images/real/'
    result_root = './result/label_result/'

    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    saliency_names = [i for i in os.listdir(saliency_root)]
    for img in saliency_names:
        # 显著性图转为灰度图
        saliency = cv2.cvtColor(cv2.imread(osp.join(saliency_root, img)), cv2.COLOR_BGR2GRAY)
        mask = np.zeros(saliency.shape)
        # 加载测试图片、归一化的同时，转为cv2.CV_64F双精度类型
        test = cv2.imread(osp.join(real_root, img[:-10] + '_real_A.png'))
        test = cv2.normalize(src=test, dst=np.zeros(test.shape), alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        saliency_np = np.array(saliency)
        if saliency_np.min() > 200:
            def_box = test
        else:
            # 计算显著图二值化的灰度阈值
            closed_thres = max(saliency_np.min()+20, 170)
            # 得到显著图的二值图saliencyBinary
            _, saliencyBinary = cv2.threshold(src=saliency, thresh=closed_thres, maxval=255, type=cv2.THRESH_BINARY_INV)
            # 形态学闭运算
            # 从样本实验效果看，开运算似乎会更好一点
            closed_mask = cv2.morphologyEx(saliencyBinary, cv2.MORPH_CLOSE, kernel5, iterations=1)
            opened_mask = cv2.morphologyEx(saliencyBinary, cv2.MORPH_OPEN, kernel5, iterations=1)
            # 形态重建：
            _, marker = cv2.threshold(src=saliency, thresh=saliency_np.min()+5, maxval=255, type=cv2.THRESH_BINARY_INV)
            # 以closed_mask为约束，不断膨胀marker。mask >= marker
            defect = imreconstruct(marker, closed_mask)
            # 测量标记的图像区域的属性，值为0的标签将被忽略（忽略黑色部分）
            # 返回一个”描述带标签的区域“列表
            stats = measure.regionprops(defect)
            for region in stats:
                # begion.bbox:边界外接框
                mask[ceil(region.bbox[0]):floor(region.bbox[2]), ceil(region.bbox[1]):floor(region.bbox[3])] = 1
            mask_seg = segImage(test, cv2.dilate(mask, kernel3).astype('float64'))
            cv2.imwrite(osp.join(result_root, img[:-10] + '_mask.png'), mask_seg);
            # cv2.imshow('inshow', mask_seg)
            # cv2.waitKey(0)

if __name__ == '__main__':
    main()
















