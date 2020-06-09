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
        tar = np.float64(tar)
        cv2.normalize(tar, dst=tar, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        # read、toFloat、 norm
        ori = cv2.imread(osp.join(img_root, patch_names[idx+2]))
        ori = np.float64(ori)
        cv2.normalize(ori, dst=ori, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        
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
        diff = np.sum(diff, axis=2)
        
        # _, diff_binary = cv2.threshold(diff, 0.05, 255, cv2.THRESH_BINARY_INV)
        # if(np.sum(diff_binary)<20):
            # def_seg = ori
            # cv2.imwrite(def_seg, ['images/', patch_names(img).name(1:end-9), '_seg.png'])
            # cv2.imwrite(osp.join('./py-result/'), def_seg)
            # print(str(img) + ': No defect detected!'])
            # continue
        
        seed_thresh = min(np.percentile(diff, 99.8), 0.2)
        grow_thresh = min(max(np.percentile(diff, ratio), 0.03), seed_thresh)
        
        _, marker = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)
        _, mask = cv2.threshold(diff, grow_thresh, 255, cv2.THRESH_BINARY)
        defect = imreconstruct(marker, mask)
        
        defect = cv2.morphologyEx(np.pad(defect, (4,4), mode='constant'), cv2.MORPH_CLOSE, kernel9, iterations=1)
        defect = defect[5:-3, 5:-3]
        
        stats = measure.regionprops(defect.astype(int))
        for region in stats:
            # measure.regionprops的返回值的属性相关介绍太少！
            # filled_image:Binary region image with filled holes which has the same size as bounding box.
            defect[ceil(region.bbox[0]):floor(region.bbox[2]), ceil(region.bbox[1]):floor(region.bbox[3])] = region.filled_image 
        # 相比于matlab-code,这里还省略了一些操作，但似乎不影响最终结果
        def_seg = segImage(ori, defect.astype('float64'));
        cv2.imwrite('./result/seg_result/'+ patch_names[idx+2], ori*255)
        # cv2.imshow('', ori)
        # cv2.waitKey(0)
        

if __name__ == '__main__':
    main()