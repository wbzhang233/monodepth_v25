import math
import cv2
import numpy as np

def calculate_vdisparity(disp_img,max_disp, img_height):
    # calculate v-disparity
    vhist_vis = np.zeros((img_height, max_disp), np.float)
    for i in range(img_height):
        vhist_vis[i, ...] = cv2.calcHist(images=[disp_img[i, ...]], channels=[0], mask=None, histSize=[max_disp],
                                         ranges=[0, max_disp]).reshape(255).flatten() / float(img_height)

    vhist_vis = np.array(vhist_vis * 255, np.uint8)
    vblack_mask = vhist_vis < 5
    vhist_vis = cv2.applyColorMap(vhist_vis, cv2.COLORMAP_JET)
    vhist_vis[vblack_mask] = 0
    return vhist_vis

def calculate_udisparity(disp_img, max_disp, img_width):
    # calculate u-disparity
    uhist_vis = np.zeros((max_disp, img_width), np.float)
    for i in range(img_width):
        uhist_vis[..., i] = cv2.calcHist(images=[disp_img[..., i]], channels=[0], mask=None, histSize=[max_disp],
                                         ranges=[0, max_disp]).reshape(255).flatten() / float(img_width)

    uhist_vis = np.array(uhist_vis * 255, np.uint8)
    ublack_mask = uhist_vis < 5
    uhist_vis = cv2.applyColorMap(uhist_vis, cv2.COLORMAP_JET)
    uhist_vis[ublack_mask] = 0
    return uhist_vis




