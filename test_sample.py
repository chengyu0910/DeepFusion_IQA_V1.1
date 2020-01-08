from PIL import Image
import numpy as np
import util.util as util
import torchvision.transforms as transforms
from models.networks import GatedNetGenerator
from data import image_folder
from data import custom_dataset
import torch
from torch.autograd import Variable
from math import ceil
from guided_filter.results.smooth_noise import GuidedFilterImg
import os
from setgpu import setgup

epoch = 'latest'
data = 'real'
loss = 'joint'
images_dir = './imgs/' + data + '/'
results_dir = './imgs/results_' + loss + '_' + data +'_'+ epoch + '/'
if os.path.exists(results_dir) is False:
    os.makedirs(results_dir)
derives_dir = './imgs/derives_' + loss + '_' + data + '_'+ epoch + '/'
if os.path.exists(derives_dir) is False:
    os.makedirs(derives_dir)


paramspath = '/home/cy/Project/DeepFusion_Experiments/DeepFusion_IQA_V1.0/checkpoints/DeepFusion_'+loss+'/'+epoch+'_net_GatedNet.pth'
setgup(6)
device = torch.device('cuda:0')
imgnames = image_folder.make_dataset(images_dir)
model = GatedNetGenerator().cuda().eval()
model.load_state_dict(torch.load(paramspath))


for i,imgname in enumerate(imgnames):
    print(imgname)
    dark = np.array(Image.open(images_dir + imgname).convert('RGB')).astype(np.float32)/255
    height, width, chs = dark.shape
    height = int(height / 8) * 8
    width = int(width / 8) * 8
    enhanced = np.zeros([height,width,chs])
    map_ch = np.zeros([height,width,chs])
    map_lc = np.zeros([height,width,chs])
    map_lg = np.zeros([height,width,chs])
    img_ch = np.zeros([height,width,chs])
    img_lc = np.zeros([height,width,chs])
    img_lg = np.zeros([height,width,chs])


    num_patchs = ceil(width/1600)

    start_h = 0
    stride_h = int(ceil(height/num_patchs)/8)*8
    while start_h < height:
        if start_h + stride_h > height:
            stride_h = height - start_h
            stride_h = int(stride_h/8)*8

        start_w = 0
        stride_w = int(ceil(width / num_patchs) / 8) * 8
        while start_w < width:
            if start_w + stride_w > width:
                stride_w = width - start_w
                stride_w = int(stride_w / 8) * 8

            dark_patch = dark[start_h:start_h+stride_h, start_w:start_w+stride_w,:]
            img_ch_patch = custom_dataset.clahe(dark_patch,3.0)
            img_ch[start_h:start_h + stride_h, start_w:start_w + stride_w,:] = img_ch_patch#TODO:numpy to Image
            img_lg_patch = custom_dataset.LogEn(dark_patch, 10)
            img_lg[start_h:start_h + stride_h, start_w:start_w + stride_w,:] = img_lg_patch
            guidance = img_lg_patch
            img_lc_patch = custom_dataset.LightEn(dark_patch, guidance)
            img_lc[start_h:start_h + stride_h, start_w:start_w + stride_w, :] = img_lc_patch

            input = torch.cat((torch.from_numpy(dark_patch.astype(np.float32).transpose((2, 0, 1))).cuda(),
                               torch.from_numpy(img_ch_patch.astype(np.float32).transpose((2, 0, 1))).cuda(),
                               torch.from_numpy(img_lc_patch.astype(np.float32).transpose((2, 0, 1))).cuda(),
                               torch.from_numpy(img_lg_patch.astype(np.float32).transpose((2, 0, 1))).cuda()),0)

            del img_ch_patch
            del img_lc_patch
            del img_lg_patch
            del dark_patch
            with torch.no_grad():
                results = model.forward(torch.unsqueeze(input, 0))
                enhanced[start_h:start_h + stride_h, start_w:start_w + stride_w,:] = np.clip(results['en'].data.squeeze().cpu().numpy(),0,1).transpose((1,2,0))
                map_ch[start_h:start_h + stride_h, start_w:start_w + stride_w,:] = np.clip(results['ch_map'].data.squeeze().cpu().numpy(),0,1).transpose((1,2,0))
                map_lc[start_h:start_h + stride_h, start_w:start_w + stride_w,:] = np.clip(results['lc_map'].data.squeeze().cpu().numpy(),0,1).transpose((1,2,0))
                map_lg[start_h:start_h + stride_h, start_w:start_w + stride_w,:] = np.clip(results['lg_map'].data.squeeze().cpu().numpy(),0,1).transpose((1,2,0))

            del input
            del results
            start_w += stride_w
        start_h += stride_h

    Image.fromarray((enhanced * 255).astype(np.uint8)).save(results_dir + imgname[:-4] + '_enhanced.bmp')
    Image.fromarray((map_ch * 255).astype(np.uint8)).save(derives_dir + imgname[:-4] + '_map_ch.bmp')
    Image.fromarray((map_lc * 255).astype(np.uint8)).save(derives_dir + imgname[:-4] + '_map_lc.bmp')
    Image.fromarray((map_lg * 255).astype(np.uint8)).save(derives_dir + imgname[:-4] + '_map_lg.bmp')

    Image.fromarray((img_ch * 255).astype(np.uint8)).save(derives_dir + imgname[:-4] + '_ch.bmp')
    Image.fromarray((img_lc * 255).astype(np.uint8)).save(derives_dir + imgname[:-4] + '_lc.bmp')
    Image.fromarray((img_lg * 255).astype(np.uint8)).save(derives_dir + imgname[:-4] + '_lg.bmp')


    del enhanced

    #guided filter process confidence maps
    #transfer to RGB,0-1,float
    print('引导滤波后处理')
    guidance = img_lg
    map_ch = GuidedFilterImg(np.float32(map_ch), guidance)
    map_lc = GuidedFilterImg(np.float32(map_lc), guidance)
    map_lg = GuidedFilterImg(np.float32(map_lg), guidance)

    enahnced_guided = np.clip(map_ch * img_ch + map_lc * img_lc + map_lg * img_lg,0,1)

    util.save_image(enahnced_guided*255, results_dir + imgname[:-4] + '_enhanced_guided.bmp')
    util.save_image(map_ch*255, derives_dir + imgname[:-4] + '_map_ch_guided.bmp')
    util.save_image(map_lc*255, derives_dir + imgname[:-4] + '_map_lc_guided.bmp')
    util.save_image(map_lg*255, derives_dir + imgname[:-4] + '_map_lg_guided.bmp')

    print('处理 %d / %d 张' % (i + 1, len(imgnames)))

