import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from util import util
from torch.autograd import Variable
import cv2
from guided_filter.results.smooth_noise import GuidedFilterImg


class GatedNetDataset(BaseDataset):
    #读出4张图片:dark,wb,ce,gm,聚合成一个tensor
    #随机裁剪128*128大小
    def initialize(self, opt, mode='train'):

        self.opt = opt
        self.mode = mode
        self.root= opt.dataroot
        if self.mode == 'train':
            self.darks_dir = os.path.join(opt.dataroot) + '/data/trainset/'
            self.deriveds_dir = os.path.join(opt.dataroot) + '/deriveds/trainset/'
        else:
            self.darks_dir = os.path.join(opt.dataroot) + '/data/testset/'
            self.deriveds_dir = os.path.join(opt.dataroot) + '/deriveds/testset/'
        self.labels_dir = os.path.join(opt.dataroot) + '/label/'
        self.dark_names = sorted(make_dataset(self.darks_dir))#make_dataset 从文件夹读出所有图片文件并返回 sorted按从小到大排序
        # if self.opt.isTrain == False:#测试条件下 抽样
        #     print('size of trainset',len(self.dark_names))
        #     self.dark_names = random.sample(self.dark_names,1000)
        #     print('size of testset',len(self.dark_names))
        transform_list = [
                          transforms.ToTensor()
                          ]#transforms是对样本做处理，image转换为tensoor
        self.transform = transforms.Compose(transform_list)#Compose真正执行处理

    def __getitem__(self, index):
        #get trans_input
        self.dark_name = self.dark_names[index]
        img_dark = self.preprocess(Image.open(self.darks_dir + self.dark_name).convert('RGB'), self.opt.fineSize, rand_new=True)#PIL读取出的是0-255范围,pixel = img_dark.load()  pixel[x,y]就是像素值
        img_label = self.preprocess(Image.open(self.labels_dir + self.dark_name.split('_')[0]).convert('RGB'), self.opt.fineSize)
        img_ch = self.preprocess(Image.open(self.deriveds_dir + self.dark_name[:-4] + '_ch.png').convert('RGB'), self.opt.fineSize)
        img_lc = self.preprocess(Image.open(self.deriveds_dir + self.dark_name[:-4] + '_lc.png').convert('RGB'), self.opt.fineSize)
        img_lg = self.preprocess(Image.open(self.deriveds_dir + self.dark_name[:-4] + '_lg.png').convert('RGB'), self.opt.fineSize)

        input = torch.cat((img_dark, img_ch, img_lc, img_lg), 0)#concate dark and deriveds to a tensor
        return {'input_concat': input, 'groundtruth': img_label, 'paths': self.dark_name}

    def preprocess(self, img, crop_size, rand_new=False):
        #img: image obtained bt Image package, Image type
        #return: preprocessed image, torch.Tensor type

        if rand_new is True:
            self.resize = random.randint(0, 90)
            self.flip = random.randint(0, 90)
            self.rotate = random.randint(0, 90)


        scalelist = np.asarray([0.5, 1.3, 1.7, 2.0, 2.5])
        resize_after_crop = False
        crop_size_original = crop_size
        if self.opt.resize is True:
            if self.resize < 30:
                scale = scalelist[self.resize%5]
                if scale*img.size[0] >= crop_size and scale*img.size[1] >= crop_size:
                    crop_size = int(crop_size/scale) + 1
                    resize_after_crop = True
                #else: do nothing, keep the original size, do not resize.

        if self.opt.crop is True:
            if crop_size <= img.size[0] and crop_size <= img.size[1]:
                if rand_new is True:
                    self.w_offset = random.randint(0, img.size[0] - crop_size )
                    self.h_offset = random.randint(0, img.size[1] - crop_size )
                img = img.crop((self.w_offset, self.h_offset, self.w_offset + crop_size, self.h_offset + crop_size))
            else:
                if rand_new is True:
                    if crop_size > img.size[0] and crop_size <= img.size[1]:
                        self.w_offset = 0
                        self.h_offset = random.randint(0, img.size[1] - img.size[0])
                    elif crop_size <= img.size[0] and crop_size > img.size[1]:
                        self.w_offset = random.randint(0, img.size[0] - img.size[1])
                        self.h_offset = 0
                    else:
                        self.w_offset = random.randint(0, img.size[0] - min(img.size[0],img.size[1]))
                        self.h_offset = random.randint(0, img.size[0] - min(img.size[0],img.size[1]))
                img = img.crop((self.w_offset, self.h_offset, self.w_offset + crop_size, self.h_offset + crop_size))
                resize_after_crop = True
                # print('image name: %s, %d, %d'%(self.dark_name, crop_size, img.size[0]))

        if resize_after_crop is True:# reduce the computation in resize phase, if resize and crop are both required.
            img = img.resize((crop_size_original, crop_size_original),
                             Image.ANTIALIAS)  # img.size[0] is width

        if self.opt.flip is True:
            if self.flip % 3 == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.flip % 3 == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.opt.rotate is True:
            if self.rotate < 30:
                img = img.rotate((self.rotate % 3 + 1)*90)

        img = self.transform(img)  # 转换成tensor，变成0-1范围

        return img

    def __len__(self):
        return len(self.dark_names)

    def name(self):
        if self.mode == 'train':
            return 'DeepFusionNet_Trainset'
        else:
            return 'DeepFusionNet_Testset'



def LogEn(img,v):
    c = 1.0
    enimg = c * np.log(1 + v * img) / np.log(v+1)
    return enimg

    # c = 1.0
    # enimg = c * torch.log(1 + v * img) / torch.log(torch.FloatTensor([v + 1]))
    # return enimg

def LightEn(img, guidance):
    lc = img.max(axis=2)
    lc = GuidedFilterImg(lc, guidance)
    lc[lc == 0] = 0.001
    rf = np.zeros(img.shape, dtype = img.dtype)
    for i in range(3):
        rf[:,:,i] = img[:,:,i]/ lc
    lc_new = img**0.8#保证图像不会有块效应，而且保持了与原图色彩一致性
    img_lc = lc_new*rf
    img_lc = np.clip(img_lc,0,1)
    return img_lc


    # lctuple = torch.max(img, dim=0)
    # lc = lctuple[0]
    #
    # if guidance is None:
    #     guidance = img.numpy().transpose((1,2,0))
    #
    # lc = lc.numpy()
    # lc = GuidedFilterImg(lc, guidance)
    # lc = torch.from_numpy(lc)
    #
    # lc[lc == 0] = 0.001/255
    # rf = img/lc
    # lc_new = img**0.6#保证图像不会有块效应，而且保持了与原图色彩一致性
    # lcimg = lc_new*rf
    # return lcimg

def clahe(img,limit):

    #tensor2numpy
    im = img*255
    im = im.astype(np.uint8)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    del im
    im_hsv_new = np.zeros(im_hsv.shape)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    im_hsv_new[:,:,0] = im_hsv[:,:,0]
    im_hsv_new[:,:,1] = im_hsv[:,:,1]
    im_hsv_new[:,:,2] = clahe.apply(im_hsv[:,:,2])
    im_new = cv2.cvtColor(im_hsv_new.astype(np.uint8), cv2.COLOR_HSV2RGB)
    im_new = im_new.astype(np.float32)/255

    return im_new

    # #tensor2numpy
    # im = img.numpy()
    # im = im*255
    # im = im.astype(np.uint8)
    # im = np.transpose(im,[1,2,0])
    # im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    # del im
    # im_hsv_new = np.zeros(im_hsv.shape)
    # clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    # im_hsv_new[:,:,0] = im_hsv[:,:,0]
    # im_hsv_new[:,:,1] = im_hsv[:,:,1]
    # im_hsv_new[:,:,2] = clahe.apply(im_hsv[:,:,2])
    # im_new = cv2.cvtColor(im_hsv_new.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # im_new = im_new.astype(np.float32)/255
    # im_new = np.transpose(im_new,[2,0,1])
    #
    # im_new = torch.from_numpy(im_new)
    # #numpy2tensor
    # return im_new


def GammaCorrected(imgsrc,gm):
    img_gm = imgsrc**gm
    return img_gm

def RealGWbal(img):#输入为0-255,输出为0-1
    r = img[0, :, :]
    g = img[1, :, :]
    b = img[2, :, :]

    avgR = torch.mean(r[:])
    avgG = torch.mean(g[:])
    avgB = torch.mean(b[:])

    avg_ch = [avgR + 0.001/255, avgG + 0.001/255, avgB + 0.001/255]
    avgRGB = (avgR + avgG + avgB)/3
    scaleValue = [avgRGB / avg_ch[0], avgRGB / avg_ch[1], avgRGB / avg_ch[2]]

    img_wb = torch.FloatTensor(img.shape)
    img_wb[0,:, :] = scaleValue[0] * r
    img_wb[1,:, :] = scaleValue[1] * g
    img_wb[2,:, :] = scaleValue[2] * b
    img_wb[img_wb > 1] = 1
    return img_wb

def ContrastEnhanced(imgsrc):
    u = torch.mean(imgsrc)
    img_ce = (2 * (0.5 + u)) * (imgsrc - u)
    return img_ce
