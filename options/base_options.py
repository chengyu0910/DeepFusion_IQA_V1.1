import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # self.parser.add_argument('--labelroot',help='path to trans_label (should have subfolders trainA, trainB, valA, valB, etc)')
        # self.parser.add_argument('--hazeroot',help='path to input_haze (should have subfolders trainA, trainB, valA, valB, etc)')
        # self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        # self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        # self.parser.add_argument('--depth_nc', type=int, default=3, help='# of depth image channels')
        # self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        # self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        # self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        # self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        # self.parser.add_argument('--which_model_depth', type=str, default='aod', help='selects model to use for generating depth')
        # self.parser.add_argument('--non_linearity', type=str, default=None, help='last nonliearity layer None(tanh)| linear | sigmoid | ReLU | BReLU')
        # self.parser.add_argument('--pooling', action='store_true',  help='if specified, use pooling layers in AODNet')
        # self.parser.add_argument('--filtering', type=str, default=None, help='filtering at the last layer: None | max')
        # self.parser.add_argument('--n_layers', type=int, default=3, help='# of layers for airlight generator')
        # self.parser.add_argument('--n_layers_D', type=int, default=5, help='only used if which_model_netD==n_layers')
        # self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        # self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        # self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        # self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        # self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        # self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        # self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        # self.parser.add_argument('--depth_reverse', action='store_true', help='if True, the matting / depth images will be reversed (1 - alpha)')
        # self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        # self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='multifusion',help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--display', action='store_true', default=True,help='whether display results and intermmediates')
        self.parser.add_argument('--torchversion', type=int, default=4,help='if True, the matting / depth images will be reversed (1 - alpha)')




        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()#控制台得到训练的opt 包括batchzie non_linear的类型等网络特性
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])#设置GPU

        args = vars(self.opt)

        print('------------ Options -------------')#得到的属性字典打印出来
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in args.items():
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
