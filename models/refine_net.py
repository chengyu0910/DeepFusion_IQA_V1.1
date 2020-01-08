import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class RefineNetModel(BaseModel):
    def name(self):
        return 'RefineNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.label = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks

        self.net = networks.define_refine(opt.isTrain,self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.net, 'refine', opt.which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.Loss1 = torch.nn.L1Loss()
            self.Loss2 = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.net)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_data = input['input']
        input_label = input['label']

        self.input.resize_(input_data.size()).copy_(input_data)#resize
        self.label.resize_(input_label.size()).copy_(input_label)
        self.image_paths = input['paths']

    def forward(self):
        self.input_data = Variable(self.input)
        self.input_label = Variable(self.label)
        self.prediction = self.net.forward(self.input_data)


    # no backprop gradients
    def test(self):
        self.input_data = Variable(self.input, volatile=True)
        self.input_label = Variable(self.label, volatile=True)#volatile=True表示该Variable不需要求导
        self.prediction = self.net.forward(self.input_data)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward(self):
        #self.Loss = 0.5*self.Loss1(self.prediction,self.input_label) + 0.5*self.Loss2(self.prediction,self.input_label)
        self.Loss = self.Loss2(self.prediction,self.input_label)
        self.l2loss = self.Loss2(self.prediction,self.input_label)
        #print('Training Error: %f '%(self.Loss.data[0]))
        self.Loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()#清空梯度
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        # print(type(self.Loss.data))
        # print(type(self.Loss.data[0]))
        #print('平均误差',self.Loss.data[0])
        loss_mean = torch.mean(self.Loss.data)
        l2_mean =  torch.mean(self.l2loss.data)
        print('L2LOSS:',l2_mean)
        return OrderedDict([('Refine_loss', loss_mean),('L2LOSS',l2_mean)])#

    def get_current_visuals(self):
        label = util.tensor2im(self.input_label.data)
        prediction = util.tensor2im(self.prediction.data)
        inputtrans = util.tensor2im(self.input_data.data)
        return OrderedDict([('label', label), ('prediction', prediction)])

    def save(self, label):#这里的label是标签的意思  其他的是groundtruth的意思
        self.save_network(self.net, 'refine', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
