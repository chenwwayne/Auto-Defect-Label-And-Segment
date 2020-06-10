import os
import os.path as osp
import cv2
import numpy as np
from skimage import measure
from segImage import segImage
from math import ceil, floor

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
        dilation = cv2.dilate(marker, kernel=SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    return marker

def main():
    img_root = './images/all/'
    patch_names = [i for i in os.listdir(img_root)]

    kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    for idx in range(0,len(patch_names)-3, 3):
        sal = cv2.imread(osp.join(img_root, patch_names[idx]))
        # read、toFloat、 norm
        tar = cv2.imread(osp.join(img_root, patch_names[idx+1]))
        tar = cv2.normalize(src=tar, dst=np.zeros(tar.shape), alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        # read、toFloat、 norm
        ori = cv2.imread(osp.join(img_root, patch_names[idx+2]))
        ori = cv2.normalize(src=ori, dst=np.zeros(ori.shape), alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        
        DefSize = patch_names[idx].split('__')[1][0]
        if DefSize == 'S':
            ratio = 99
        elif DefSize == 'M':
            ratio = 98
        elif DefSize == 'L':
            ratio = 97
        else:
            ratio = 95
        
        diff = np.array((ori-tar) ** 2)
        diff = np.sum(diff, axis=2) # 三个通道合为一个通道

        _, diff_binary = cv2.threshold(diff, 0.05, 255, cv2.THRESH_BINARY)
        # 二值图几乎都是黑色
        if(np.sum(diff_binary)<20):
            def_seg = ori
            # cv2.imwrite(osp.join('./py-result/'), def_seg)
            print(str(img) + ': No defect detected!')
            continue
        
        # 这里求分位数的结果与matlab有点不一样
        # diff中白色（1）为缺陷，黑色为背景（0）
        # np.percentile(diff, 99.8)即找到排在99.8%位置的像素值,也即找到最亮的像素值
        seed_thresh = min(np.percentile(diff, 99.8), 0.2)
        grow_thresh = min(max(np.percentile(diff, ratio), 0.03), seed_thresh)

        _, marker = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)
        _, mask = cv2.threshold(diff, grow_thresh, 255, cv2.THRESH_BINARY)
        # 区域生长的意义在什么地方？
        defect = imreconstruct(marker, mask)
        
        defect = cv2.morphologyEx(np.pad(defect, (4,4), mode='constant'), cv2.MORPH_CLOSE, kernel9, iterations=1)
        defect = defect[5:-3, 5:-3]
        
        stats = measure.regionprops(defect.astype(int))
        for region in stats:
            # measure.regionprops的返回值的属性相关介绍太少！
            # filled_image:Binary region image with filled holes which has the same size as bounding box.
            # https://www.jianshu.com/p/1b90b549b50e
            defect[ceil(region.bbox[0]):floor(region.bbox[2]), ceil(region.bbox[1]):floor(region.bbox[3])] = region.filled_image 
            
            
            # ========这里有待验证正确性==========
            if region.area < 10: # 如果缺陷面积小于10个像素，则填成黑色
                for xy in region.coords:
                    defect[xy[0], xy[1]] = 0
            # ========这里有待验证正确性==========

        def_seg = segImage(ori, defect.astype('float64'));
        # cv2.imwrite('./py-result/'+ patch_names[idx+2], ori*255)
        cv2.imshow('', def_seg)
        cv2.waitKey(0)
        

if __name__ == '__main__':
    main()