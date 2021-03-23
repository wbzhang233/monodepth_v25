'''
用于根据相对深度图计算视差图，uv视差检测障碍物
'''

import uvdisp
import uvdisp2
import cv2
import numpy as np
import matplotlib.pyplot as plt

def limitMax(npMat,vmax):
    npMat[npMat>vmax] = vmax
    return npMat

# 归一化
def normalization(npMat):
    _range = np.max(npMat)-np.min(npMat)
    return (npMat-np.min(npMat)) / _range

# 标准化
def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data-mu)/sigma

if __name__ == "__main__":
    for i in range(0, 26):
        file_name = '/media/1417-1859/Datasets/sitl_datasets/mono/monodepth2_5_results/npy/frame%04d_disp.npy' % (i + 1)
        image = np.load(file_name)
        image = np.squeeze(image)

        # plt.imshow(image)
        # cv2.imwrite(str(i)+'.png',image)
        # plt.show()
        cv2.normalize(image, image, 1.0, 0.0, cv2.NORM_MINMAX)
        # 高斯滤波
        # image = cv2.GaussianBlur(image,(5,5),1.0,None,1.0,cv2.BORDER_DEFAULT)
        # image = cv2.medianBlur(image,3)
        image = cv2.bilateralFilter(image,9,150,150)
        cv2.imshow('vis', image)
        # cv2.imwrite('./obs5/vis/' + str(i) + '.png', image)

        rel_image = np.uint8(normalization(image)*255)
        h,w = image.shape

        # 根据相对深度图计算相对视差图
        ones = np.ones(image.shape)
        disp = np.uint8(np.divide(255,image))
        vmax = disp.max()*0.95
        disp = limitMax(disp,vmax)
        disp = np.uint8(normalization(disp)*255)
        cv2.imshow('disp',disp)
        # cv2.waitKey(0)

        # 根据相对视差图计算UV视差图
        scaling_factor = 255

        'uvdisp 计算uv视差'
        u_disparity = uvdisp.calculate_udisparity(disp_img=disp, max_disp=scaling_factor, img_width=w)
        v_disparity = uvdisp.calculate_vdisparity(disp_img=disp, max_disp=scaling_factor, img_height=h)

        'uvdisp2 计算障碍物'
        # obs_u = np.zeros((h, w), np.uint8)
        # obs_v = np.zeros((h, w), np.uint8)
        # obs = np.zeros((h, w), np.uint8)
        # u_disparity = uvdisp2.calculate_udisparity(disp_img=rel_image, max_disparity=scaling_factor, img_height=h,
        #                                    img_width=w, obstacle=obs_u)
        # v_disparity = uvdisp2.calculate_vdisparity(disp_img=rel_image, max_disparity=scaling_factor, img_height=h,
        #                                    img_width=w, obstacle=obs_v)

        cv2.imshow('udisp', u_disparity)
        cv2.imshow('vdisp', v_disparity)

        if i == 25:
            cv2.waitKey()
        else:
            cv2.waitKey(10)

    print('All images done.')