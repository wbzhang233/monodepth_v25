import numpy as np
import cv2
import matplotlib.pyplot as plt


for i in range(1,28):
    file_name = '/media/1417-1859/Datasets/sitl_datasets/mono/monodepth2_5_results/npy/frame%04d_disp.npy'%(i+1)
    image = np.load(file_name)
    # plt.imshow(image[0,0,:,:])
    # cv2.imwrite(str(i)+'.png',image[i,0,:,:])
    # plt.show()

    pic = image[0,0,:,:]
    cv2.normalize(pic,pic,1.0,0.0,cv2.NORM_MINMAX)
    cv2.imshow('vis',pic)
    # cv2.imwrite('/media/1417-1859/Datasets/sitl_datasets/mono/monodepth2_5_results/vis/'+str(i)+'.png',pic)
    if i==26:
        cv2.waitKey()
    else:
        cv2.waitKey(10)
