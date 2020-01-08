import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class GatedCoarsestNetModel(BaseModel):
    def name(self):
        return 'GatedCoarsestNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.groundtruth = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        # load/define networks
        self.net = networks.define_GatedCoarsestNet(opt.isTrain,self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.net, 'GatedCoarsestNet', opt.which_epoch)

        #define loss ,optimzer
        if self.isTrain:
            self.base_lr = opt.lr
            # define loss functions
            self.mseloss = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.00001)

        print('----------- GatedCoarsestNet initialized --------------')
        networks.print_network(self.net)
        print('-------------------------------------------------------')

    def set_input(self, input):
        #input_data = input['input_concat']
        #input_label = input['groundtruth']
        self.input.resize_(input['input_concat'].size()).copy_(input['input_concat'])#resize
        self.groundtruth.resize_(input['groundtruth'].size()).copy_(input['groundtruth'])
        self.image_paths = input['paths']

    def forward(self):
        #转换成Variable类型，输入网络
        self.input_var = Variable(self.input)
        self.groundtruth_var = Variable(self.groundtruth)
        self.dehaze = self.net.forward(self.input_var)


    # no backprop gradients
    def test(self):
        self.input_var = Variable(self.input, volatile=True)
        self.groundtruth_var = Variable(self.groundtruth, volatile=True)#volatile=True表示该Variable不需要求导
        self.dehaze = self.net.forward(self.input_var)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward(self):
        self.Loss = self.mseloss(self.dehaze,self.groundtruth_var)
        self.Loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()#清空梯度
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        loss_mean = torch.mean(self.Loss.data)
        print('L2LOSS:', loss_mean)
        return OrderedDict([('L2LOSS', loss_mean)])#

    def get_current_visuals(self):
        groundtruth = util.tensor2im(self.groundtruth_var.data)
        dehaze = util.tensor2im(self.dehaze.data)
        return OrderedDict([('label', groundtruth), ('prediction', dehaze)])

    def save(self, iters):
        self.save_network(self.net, 'GatedCoarsestNet', iters, self.gpu_ids)

    def update_learning_rate(self, iters, maxiter):#每10000次迭代更新一次

        lr = self.base_lr * 0.25
        self.base_lr = lr
        print('update learning rate: %f' % (lr))
        # lr = self.base_lr * ((1 - (iters/maxiter))^2)
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = lr
        # print('update learning rate: %f' % (lr))

