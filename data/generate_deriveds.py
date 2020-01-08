#coding=utf-8
import time
from PIL import Image
from data.image_folder import make_dataset
import data.custom_dataset as pre_process
import numpy as np
from util import clock
import os

if __name__ == '__main__':
    os.chdir('/home/cy/Project/DeepFusion_Experiments/DeepFusion_IQA_V1.0_copy/')
    derives_save_dir = '/data0/data_cy/DeepFusion_dataset_agml/derives_lc5/testset/'

    if not os.path.exists(derives_save_dir):
        os.makedirs(derives_save_dir)

    dark_dir = '/data0/data_cy/DeepFusion_dataset_agml/data/testset/'
    dark_names = sorted(make_dataset(dark_dir))
    num = len(dark_names)
    for i, imgname in enumerate(dark_names):
        clock.tic()
        imghaze = Image.open(dark_dir + imgname).convert('RGB')
        imghaze = np.array(imghaze, dtype=np.float32)/255
        #derived images
        ch_input = pre_process.clahe(imghaze,3.0)
        lg_input = pre_process.LogEn(imghaze, 10)
        lc_input = pre_process.LightEn(imghaze,lg_input)

        Image.fromarray((ch_input * 255).astype(np.uint8)).save(derives_save_dir + imgname[:-4] + '_ch.png')
        Image.fromarray((lg_input * 255).astype(np.uint8)).save(derives_save_dir + imgname[:-4] + '_lg.png')
        Image.fromarray((lc_input * 255).astype(np.uint8)).save(derives_save_dir + imgname[:-4] + '_lc.png')

        t = clock.toc(silence=True)
        rt = t * (num-i)
        print('process use: %f s, rest: %f min, %d/%d'%(t,rt/60,i,num))

    print('Finished!')

