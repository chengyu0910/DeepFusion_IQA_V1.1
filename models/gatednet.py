import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
from pytorch_msssim import ms_ssim
import ast


class DeepFusionNet(BaseModel):
    def name(self):
        return 'DeepFusionNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.loss_weights = ast.literal_eval(self.opt.loss_weights)

        self.iqa_valid = False
        for k,v in self.loss_weights.items():
            if 'iqa' in k and v is not 0:
                self.iqa_valid = True

        # define tensors
        self.input = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.groundtruth = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        if self.iqa_valid is True:
            self.net, self.iqanet = networks.define_GatedNet(self.opt.iqa_param_path,self.opt.input_nc,self.opt.phase,self.opt.gpu_ids,self.opt.torchversion)
        else:
            self.net, _ = networks.define_GatedNet(self.opt.iqa_param_path, self.opt.input_nc, self.opt.phase,self.opt.gpu_ids, self.opt.torchversion)

        if self.opt.phase is 'test' or self.opt.restore_train:
            self.load_network(self.net, 'DeepFusionNet', self.opt.which_epoch)

        #define loss ,optimzer
        if self.opt.phase is 'train':
            self.lr_cur = self.opt.init_lr
            # define loss functions
            self.mseloss = torch.nn.MSELoss()
            self.l1loss = torch.nn.L1Loss()
            self.ssimloss = ms_ssim
            self.cross_entrpy = torch.nn.CrossEntropyLoss(),

            if len(self.opt.gpu_ids) > 0:
                if self.opt.torchversion == 3:
                    self.mseloss = self.mseloss.cuda(0)
                    self.l1loss = self.l1loss.cuda(0)
                else:
                    self.device = torch.device('cuda')
                    self.mseloss = self.mseloss.to(self.device)
                    self.l1loss = self.l1loss.to(self.device)
                # initialize optimizers
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=self.opt.init_lr, betas=(0.9, 0.999), weight_decay=0.00001)


        print('--------------- GatedNet initialized ------------------')
        networks.print_network(self.net)
        print('-------------------------------------------------------')

    def set_input(self, input):
        self.input.resize_(input['input_concat'].size()).copy_(input['input_concat'])#resize
        self.groundtruth.resize_(input['groundtruth'].size()).copy_(input['groundtruth'])
        self.image_paths = input['paths']

    def forward(self):
        #转换成Variable类型，输入网络
        self.input_var = Variable(self.input)
        self.gt_var = Variable(self.groundtruth)
        self.output = self.net.forward(self.input_var)
        if self.iqa_valid:
            self.foward_iqa()
        return self.output

    def foward_iqa(self):
        self.en_iqa = self.iqanet(self.output['en'])
        self.gt_iqa = self.iqanet(self.gt_var)

    # no backprop gradients
    def test(self):
        if self.opt.torchversion == 3:
            self.forward()
            self.calcu_loss()
        else:
            with torch.no_grad():
                self.forward()
                self.calcu_loss()

        return self.losslist['mse'],self.losslist['ssim']

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def calcu_loss(self):
        self.losslist = {'mse': self.mseloss(self.output['en'], self.gt_var),
                         'l1': self.l1loss(self.output['en'], self.gt_var),
                         'ssim': self.ssimloss(self.output['en'], self.gt_var)}
        if self.iqa_valid:
            iqalosslist = {'iqa_c1': self.mseloss(self.en_iqa['conv1'], self.gt_iqa['conv1']),
                           'iqa_c2': self.mseloss(self.en_iqa['conv2'], self.gt_iqa['conv2']),
                           'iqa_c3': self.mseloss(self.en_iqa['conv3'], self.gt_iqa['conv3']),
                           'iqa_c4': self.mseloss(self.en_iqa['conv4'], self.gt_iqa['conv4']),
                           'iqa_sc': self.mseloss(self.en_iqa['score'], self.gt_iqa['score'])}
            self.losslist = dict(self.losslist, **iqalosslist)

        self.Loss = 0
        self.iqaloss = torch.tensor([0]).float().to(self.device)
        for k, v in self.losslist.items():
            # print('%s: %f ' % (k, v*loss_w[k]))
            self.Loss += self.loss_weights[k] * v
            if 'iqa' in k:
                self.iqaloss.data += self.loss_weights[k] * v

    def backward(self):
        self.calcu_loss()
        self.Loss.backward()

    def optimize_parameters(self):
        if self.opt.torchversion != 3:
            torch.set_grad_enabled(True)
        self.forward()
        self.optimizer.zero_grad()#清空梯度
        self.backward()
        #梯度裁剪
        if not self.opt.grad_clip is -1:
            if self.opt.torchversion == 3:
                nn.utils.clip_grad_norm(self.net.parameters(), self.opt.grad_clip)
            else:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.grad_clip)
        self.optimizer.step()

    def get_current_errors(self):
        # loss_mean = torch.mean(self.Loss.data)
        # finestloss = loss_mean# no multi-scale structure
        return OrderedDict([('totalloss',torch.mean(self.Loss.data).cpu().numpy()),('iqaloss',torch.mean(self.iqaloss.data).cpu().numpy())] )#OrderedDict([('totalloss', loss_mean)])#

    def get_current_visuals(self):
        dark = util.tensor2img(self.input_var.data[0, 0:3, :, :])
        img_ch = util.tensor2img(self.input_var.data[0, 3:6, :, :])
        img_lc = util.tensor2img(self.input_var.data[0, 6:9, :, :])
        img_lg = util.tensor2img(self.input_var.data[0, 9:12, :, :])
        groundtruth = util.tensor2img(self.gt_var.data[0, :, :, :])

        enhanced = util.tensor2img(self.output['en'].data[0, :, :, :])
        ch_map = util.tensor2img(self.output['ch_map'].data[0, :, :, :])
        lc_map = util.tensor2img(self.output['lc_map'].data[0, :, :, :])
        lg_map = util.tensor2img(self.output['lg_map'].data[0, :, :, :])
        return OrderedDict([('ch', img_ch), ('lc', img_lc),('lg', img_lg),
                             ('ch_map', ch_map),('lc_map', lc_map),('lg_map', lg_map),
                            ('dark', dark),('enahanced', enhanced), ('groundtruth', groundtruth)])

    def save(self, iters):
        self.save_network(self.net, self.name(), iters, self.gpu_ids)

    def update_learning_rate(self, lr_new):#每10000次迭代更新一次
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_new
        print('更新学习率: %f -> %f' %(self.lr_cur,lr_new))
        self.lr_cur = lr_new
    def get_learning_rate(self):
        return self.lr_cur


