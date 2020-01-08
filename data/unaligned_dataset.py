import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random
import numpy as np


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'haze')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'clear')

        self.A_paths ,self.Aname = make_dataset(self.dir_A)
        self.B_paths ,self.Bname = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        # if opt.isTrain:
        #     random.shuffle(self.B_paths)    # ensure it is unaligned in training time
        # else:
        #     self.B_paths = sorted(self.B_paths)
        self.B_paths = sorted(self.B_paths)



        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform(opt)#图片预处理

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        #B_path = self.B_paths[index % self.B_size]#clear和haze 1:35的比例
        B_path = self.B_paths[(int(index/35)) % self.B_size]  # clear和haze 1:35的比例

        A_name = self.Aname[index]
        B_name = self.Bname[(int(index/35)) % self.B_size]

        #确保是一对样本
        print('A:%s,B:%s'%(A_name,B_name))
        assert (A_name[0:len(A_name)-4] == A_name[0:len(A_name)-4])



        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
