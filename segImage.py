import numpy as np
from math import ceil
import cv2

"""
Given an image and its segmentation mask, plot the segment boundaries
input: input image
mask: input mask, noted that mask should be binary image whose shape identity with input image
return : return a numpy array with the boundary drawn
"""
def segImage(input, mask):
    boundaryWidth = 1
    boundaryColor = 'red'
    
    mask_np = np.array(mask, dtype=np.float64)
    # 对二值Mask求导
    [cx, cy] = np.gradient(mask_np)
    # ccc为二值矩阵，导数不为0的地方即是边界
    ccc = abs(cx)+abs(cy) != 0
    
    # if boundaryWidth > 1:
        # boundaryWidth = ceil(boundaryWidth);
        # dilateWindow = np.ones(boundaryWidth, boundaryWidth)
        # ccc = cv2.imdilate(ccc, dilateWindow)
    # elif boundaryWidth < 1:
        # print('boundaryWidth has been reset to 1.')
    
    if boundaryColor == 'red':
        input[:,:,0] = np.minimum(input[:,:,0],~ccc) #B
        input[:,:,1] = np.minimum(input[:,:,1],~ccc) #G
        input[:,:,2] = np.maximum(input[:,:,2],ccc)  #R
    # elif boundaryColor =='black':
        # I(:,:,1) = min(I(:,:,1),~ccc);
        # I(:,:,2) = min(I(:,:,2),~ccc);
        # I(:,:,3) = min(I(:,:,3),~ccc);
    # else:
        # print('Does not recognize boundaryColor other than red and black');
        # I(:,:,1) = max(I(:,:,1),~ccc);
        # I(:,:,2) = max(I(:,:,2),~ccc);
        # I(:,:,3) = max(I(:,:,3),~ccc);
    return np.uint8(input*255)
    

if __name__ == '__main__':
    pass