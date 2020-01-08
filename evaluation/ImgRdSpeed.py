
import numpy as np
import os
import imageio
import time
import cv2
from PIL import Image






img_dir = '/data0/data_cy/DeepFusion_dataset_agml/data/trainset/'
img_names = os.listdir(img_dir)

t_misc_jpg = 0
t_misc_png = 0
t_misc_bmp = 0
t_cv2_jpg = 0
t_cv2_png = 0
t_cv2_bmp = 0
t_Image_jpg = 0
t_Image_png = 0
t_Image_bmp = 0

t_misc = 0
t_cv2 = 0
t_Image = 0
#image obtained by three ways must convert to numpy array
for i,img_name in enumerate(img_names):
    t = time.time()
    img1 = imageio.imread(img_dir+img_name)
    t_misc += time.time() - t

    t = time.time()
    img2 = cv2.imread(img_dir+img_name)
    t_cv2 += time.time() - t

    t = time.time()
    img3 = np.array(Image.open(img_dir+img_name))
    t_Image += time.time() - t
    print('%d/%d'%(i,len(img_names)))

print('misc: %f, cv2: %f, Image: %f'%(t_misc,t_cv2,t_Image))