import math
import cv2
import numpy as np

def calculate_vdisparity(disp_img, max_disparity, img_height, img_width, obstacle=None):
    # calculate v-disparity
    vhist_vis = np.zeros((img_height, max_disparity), np.float)
    for i in range(img_height):
        # [disp_img[i, ...]] make two dimesion arrays (just like orginal image but we just put only one row of the image here)
        # read more about calcHist at https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html
        # flatten the array to make it to one dimension array
        # divide all row with img_width value to normalize the histogram (make it less or equal to 1)
        vhist_vis[i, ...] = cv2.calcHist(images=[disp_img[i, ...]], channels=[0], mask=None, histSize=[max_disparity],
                                         ranges=[0, max_disparity]).flatten() / float(img_width)

    vhist_vis = np.array(vhist_vis * 255, np.uint8)
    # mask_threshold = max_disparity/10
    mask_threshold = 15
    vblack_mask = vhist_vis < mask_threshold
    vwhite_mask = vhist_vis >= mask_threshold

    vhist_vis[vblack_mask] = 0
    vhist_vis[vwhite_mask] = 255

    # Add houghman line extract
    lines = cv2.HoughLinesP(vhist_vis, 1, math.pi / 180.0, 5, np.array([]), 40, 10)
    a, b, c = lines.shape
    tmp = np.zeros((img_height, max_disparity), np.float)
    for i in range(a):
        # line on x1 y1 and x2 y2
        # x is disparity and y is row
        # print("{} {} {} {}".format(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]))
        cv2.line(tmp, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 0, 0), 1, cv2.LINE_AA)

        if obstacle is None:
            continue

        if lines[i][0][0] == lines[i][0][2] and lines[i][0][0] != 0:
            expect_disp = lines[i][0][0]
            # print(expect_disp)
            # for j in range row y1 to row y2
            r1 = lines[i][0][1] if lines[i][0][1] < lines[i][0][3] else lines[i][0][3]
            r2 = lines[i][0][3] if lines[i][0][3] > lines[i][0][1] else lines[i][0][1]
            for j in range(r1, r2):
                for k in range(img_width):
                    if disp_img[j][k] == expect_disp:
                        obstacle[j][k] = 125

    vhist_vis = cv2.applyColorMap(vhist_vis, cv2.COLORMAP_JET)
    # return vhist_vis
    return tmp


def calculate_udisparity(disp_img, max_disparity, img_height, img_width, obstacle=None):
    # calculate u-disparity
    uhist_vis = np.zeros((max_disparity, img_width), np.float)
    for i in range(img_width):
        uhist_vis[..., i] = cv2.calcHist(images=[disp_img[..., i]], channels=[0], mask=None, histSize=[max_disparity],
                                         ranges=[0, max_disparity]).flatten() / float(img_height)

    uhist_vis = np.array(uhist_vis * 255, np.uint8)
    mask_threshold = 10
    ublack_mask = uhist_vis < mask_threshold
    uwhite_mask = uhist_vis >= mask_threshold

    uhist_vis[ublack_mask] = 0
    uhist_vis[uwhite_mask] = 255

    # Add houghman line extract
    lines = cv2.HoughLinesP(uhist_vis, 1, math.pi / 180.0, 5, np.array([]), 40, 10)
    a, b, c = lines.shape
    tmp = np.zeros((max_disparity, img_width), np.float)
    for i in range(a):
        cv2.line(tmp, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 0, 0), 1, cv2.LINE_AA)
        # print("{} {} {} {}".format(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]))
        # x is column y is disparity
        if obstacle is None:
            continue

        if lines[i][0][1] == lines[i][0][3] and lines[i][0][1] != 0:
            expect_disp = lines[i][0][1]
            # from range column x1 to column x2
            for k in range(lines[i][0][0], lines[i][0][2] + 1):
                for j in range(img_height):
                    if disp_img[j][k] == expect_disp:
                        obstacle[j][k] = 125

    uhist_vis = cv2.applyColorMap(uhist_vis, cv2.COLORMAP_JET)
    # return uhist_vis
    return tmp