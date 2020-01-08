import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
# from pytorch_gdn import GDN
# import deepdish as dd
from scipy import misc
from scipy import stats
from models.TrancatedIQA import IQANet_trancated
###############################################################################
# Functions
###############################################################################


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         # m.weight.data.normal_(0.0, 0.02)
#         # if hasattr(m.bias, 'data'):
#         #     m.bias.data.fill_(0)
#         nn.init.xavier_uniform(m.weight.data)
#         if hasattr(m.bias, 'data'):
#             m.bias.data.fill_(0)
#     elif classname.find('BatchNorm2d') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#
#
# def get_norm_layer(norm_type='instance'):
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm)
#     return norm_layer


# def define_VGG(pretrained=True, gpu_ids=[]):
#     vgg = Vgg16()
#
#     if pretrained:
#         vgg.load_state_dict(torch.load('models/vgg16.weight'))
#
#     if len(gpu_ids) > 0:
#         vgg.cuda(device_id=gpu_ids[0])
#     return vgg
#
#
# def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], non_linearity=None, pooling=False, n_layers=3, filtering=None):
#     netG = None
#     use_gpu = len(gpu_ids) > 0
#     norm_layer = get_norm_layer(norm_type=norm)
#
#     if use_gpu:
#         assert(torch.cuda.is_available())
#
#     if which_model_netG == 'resnet_9blocks':
#         netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
#     elif which_model_netG == 'resnet_6blocks':
#         netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
#     elif which_model_netG == 'unet_128':
#         netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, non_linearity=non_linearity, filtering=filtering)
#     elif which_model_netG == 'unet_256':
#         netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, non_linearity=non_linearity, filtering=filtering)
#     elif which_model_netG == 'aod':
#         netG = AODNetGenerator(input_nc, output_nc, ngf, gpu_ids=gpu_ids, non_linearity=non_linearity, pooling=pooling, filtering=filtering, norm_layer=norm_layer)
#     elif which_model_netG == 'air':
#         netG = AirGenerator(gpu_ids=gpu_ids, n_layers=n_layers)
#     elif which_model_netG == 'resnet6_depth':
#         netG = ResnetDepthGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids, non_linearity=non_linearity, filtering=filtering)
#     elif which_model_netG == 'resnet9_depth':
#         netG = ResnetDepthGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids, non_linearity=non_linearity, filtering=filtering)
#     else:
#         raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
#     if len(gpu_ids) > 0:
#         netG.cuda(gpu_ids[0])
#         #netG.cuda(device_id=gpu_ids[0])
#     netG.apply(weights_init)
#     return netG
#
#
# def define_D(input_nc, ndf, which_model_netD,
#              n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
#     netD = None
#     use_gpu = len(gpu_ids) > 0
#     norm_layer = get_norm_layer(norm_type=norm)
#
#     if use_gpu:
#         assert(torch.cuda.is_available())
#     if which_model_netD == 'basic':
#         netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
#     elif which_model_netD == 'n_layers':
#         netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
#     elif which_model_netD == 'multi':
#         netD = MultiDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
#     else:
#         raise NotImplementedError('Discriminator model name [%s] is not recognized' %
#                                   which_model_netD)
#     if use_gpu:
#         netD.cuda(gpu_ids[0])
#     netD.apply(weights_init)
#     return netD
#
#
# def define_refine(isTrain = True,gpu_ids=[]):
#     netRefine = None
#     use_gpu = len(gpu_ids) > 0
#     if use_gpu:
#         assert(torch.cuda.is_available())
#     netRefine=RefineNetGenerator();
#     if use_gpu:
#         netRefine.cuda(gpu_ids[0])
#     netRefine.apply(weights_init)#参数初始化
#
#     if isTrain == False:
#         print('测试模式')
#         netRefine.eval()
#
#     return netRefine
#
# def define_GatedCoarsestNet(isTrain = True,gpu_ids=[]):
#     GatedCoarsestNet = None
#     use_gpu = len(gpu_ids) > 0
#     if use_gpu:
#         assert(torch.cuda.is_available())
#     GatedCoarsestNet=GatedCoarsestNetGenerator();
#     if use_gpu:
#         GatedCoarsestNet.cuda(gpu_ids[0])
#     GatedCoarsestNet.apply(weights_init)#参数初始化
#
#     if isTrain == False:
#         print('测试模式')
#         GatedCoarsestNet.eval()
#
#     return GatedCoarsestNet

def define_GatedNet(iqanet_path,input_nc, phase, gpu_ids,ver=3):
    GatedNet = GatedNetGenerator(input_nc=input_nc)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if ver==3:
            GatedNet.cuda(gpu_ids[0])
        else:
            device = torch.device('cuda')
            GatedNet.to(device)
            IQANet = IQANet_trancated(iqanet_path).to(device)#param init is done in IQANetGenerator

        GatedNet.apply(weights_init)#参数初始化
    else:
        IQANet = IQANet_trancated(iqanet_path).to(torch.device('cpu'))#

    if phase == 'test':
        print('测试模式')
        GatedNet.eval()
        IQANet.eval()

    return GatedNet, IQANet

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class TVLoss(nn.Module):
    '''
    Define Total Variance Loss for images
    which is used for smoothness regularization
    '''

    def __init__(self):
        super(TVLoss, self).__init__()

    def __call__(self, input):
        # Tensor with shape (n_Batch, C, H, W)
        origin = input[:, :, :-1, :-1]
        right = input[:, :, :-1, 1:]
        down = input[:, :, 1:, :-1]

        tv = torch.mean(torch.abs(origin-right)) + torch.mean(torch.abs(origin-down))
        return tv * 0.5


# # Defines the GAN loss which uses either LSGAN or the regular GAN.
# # When LSGAN is used, it is basically same as MSELoss,
# # but it abstracts away the need to create the target label tensor
# # that has the same size as the input
# class GANLoss(nn.Module):
#     def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
#                  tensor=torch.FloatTensor):
#         super(GANLoss, self).__init__()
#         self.real_label = target_real_label
#         self.fake_label = target_fake_label
#         self.real_label_var = None
#         self.fake_label_var = None
#         self.Tensor = tensor
#         if use_lsgan:
#             self.loss = nn.MSELoss()
#         else:
#             self.loss = nn.BCELoss()
#
#     def get_target_tensor(self, input, target_is_real):
#         target_tensor = None
#         if target_is_real:
#             create_label = ((self.real_label_var is None) or
#                             (self.real_label_var.numel() != input.numel()))
#             if create_label:
#                 real_tensor = self.Tensor(input.size()).fill_(self.real_label)
#                 self.real_label_var = Variable(real_tensor, requires_grad=False)
#             target_tensor = self.real_label_var
#         else:
#             create_label = ((self.fake_label_var is None) or
#                             (self.fake_label_var.numel() != input.numel()))
#             if create_label:
#                 fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
#                 self.fake_label_var = Variable(fake_tensor, requires_grad=False)
#             target_tensor = self.fake_label_var
#         return target_tensor
#
#     def __call__(self, input, target_is_real):
#         target_tensor = self.get_target_tensor(input, target_is_real)
#         return self.loss(input, target_tensor)
#
#
# # Define an airlight Net
# class AirGenerator(nn.Module):
#     def __init__(self, gpu_ids=[], n_layers=2):
#         super(AirGenerator, self).__init__()
#         self.gpu_ids = gpu_ids
#         self.n_layers = n_layers
#
#         model = []
#         for i in range(n_layers-1):
#             model += [nn.ReflectionPad2d(1),
#                       nn.Conv2d(3, 3, kernel_size=3, padding=0),
#                       nn.BatchNorm2d(3),
#                       nn.ReLU(True)]
#
#             model += [nn.MaxPool2d(kernel_size=2, padding=0)]
#
#         # global pooling at the last layer
#         model += [nn.ReflectionPad2d(1),
#                   nn.Conv2d(3, 3, kernel_size=3, padding=0),
#                   nn.Sigmoid(),
#                   nn.AdaptiveMaxPool2d(1)]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#             return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#         else:
#             return self.model(input)


# # Define AOD-Net
# class AODNetGenerator(nn.Module):
#     def __init__(self, input_nc=3, output_nc=1, ngf=6, norm_layer=nn.BatchNorm2d, gpu_ids=[], non_linearity=None, pooling=False, filtering=None, r=10, eps=1e-3):
#         super(AODNetGenerator, self).__init__()
#         self.input_nc = input_nc
#         self.gpu_ids = gpu_ids
#         self.pooling = pooling
#         self.filtering = filtering
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         self.non_linearity = non_linearity
#         if non_linearity is None:
#             last_act = nn.Tanh()
#         elif non_linearity == 'BReLU':
#             last_act = BReLU(0.95,0.05,0.95,0.05,True)
#         elif non_linearity == 'ReLU':
#             last_act = nn.ReLU(True)
#         elif non_linearity == 'sigmoid':
#             last_act = nn.Sigmoid()
#         elif non_linearity == 'linear':
#             last_act = None
#         else:
#             print(non_linearity)
#             raise NotImplementedError
#
#         model = [nn.Conv2d(input_nc, ngf, kernel_size=1, padding=0, bias=use_bias),
# #                 nn.BatchNorm2d(ngf),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         if pooling:
#             model += [nn.MaxPool2d(kernel_size=2, padding=0),
#                       nn.Upsample(scale_factor=2, mode='nearest')]
#
#         model += [ConcatBlock(ngf, pooling, norm_layer)]
#
#         model += [nn.ReflectionPad2d(1),
#                   nn.Conv2d(4*ngf, output_nc, kernel_size=3, padding=0)]
#
#         if last_act is not None:
#             model += [last_act]
#         self.model = nn.Sequential(*model)    # nn.Sequential only accepts a single input.
#
#         if filtering is not None:
#             if filtering == 'max':
#                 self.last_layer = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
#             elif filtering == 'guided':
#                 self.last_layer = GuidedFilter(r=r, eps=eps)
#
#     def forward(self, input):
#         if self.filtering == 'guided':
#             # rgb2gray
#             guidance = 0.2989 * input[:,0,:,:] + 0.5870 * input[:,1,:,:] + 0.1140 * input[:,2,:,:]
#             # rescale to [0,1]
#             guidance = (guidance + 1) / 2
#             guidance = torch.unsqueeze(guidance, dim=1)
#
#         if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#             pre_filter = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#             if self.non_linearity is None:
#                 # rescale to [0,1]
#                 pre_filter = (pre_filter + 1) / 2
#
#             if self.filtering is not None:
#                 if self.filtering == 'guided':
#                     return pre_filter, self.last_layer(guidance, pre_filter)
#                 else:
#                     return pre_filter, nn.parallel.data_parallel(self.last_layer, pre_filter, self.gpu_ids)
#             else:
#                 return None, pre_filter
#         else:
#             pre_filter = self.model(input)
#             if self.non_linearity is None:
#                 # rescale to [0,1]
#                 pre_filter = (pre_filter + 1) / 2
#
#             if self.filtering is not None:
#                 if self.filtering == 'guided':
#                     return pre_filter, self.last_layer(guidance, pre_filter)
#                 else:
#                     return pre_filter, self.last_layer(pre_filter)
#             else:
#                 return None, pre_filter

# Guided image filtering for grayscale images
class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3, tensor=torch.cuda.FloatTensor):    # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.tensor = tensor

        self.boxfilter = nn.AvgPool2d(kernel_size=2*self.r+1, stride=1,padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """
        
        N = self.boxfilter(Variable(self.tensor(p.size()).fill_(1),requires_grad=False))

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I*p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I*I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b

#
# class ConcatBlock(nn.Module):
#     def __init__(self, input_nc, pooling=False, norm_layer=nn.BatchNorm2d):
#         super(ConcatBlock, self).__init__()
#         self.conv_block1 = self.build_block(input_nc, 3, input_nc, pooling, norm_layer)
#         self.conv_block2 = self.build_block(2*input_nc, 5, input_nc, pooling, norm_layer)
#         self.conv_block3 = self.build_block(2*input_nc, 7, input_nc, pooling, norm_layer)
#
#     def build_block(self, input_nc, kernel_size, output_nc, pooling, norm_layer):
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         model = [nn.ReflectionPad2d(kernel_size/2),
#                  nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
#                  norm_layer(output_nc),
#                  nn.ReLU(True)]
#
#         if pooling:
#             model += [nn.MaxPool2d(kernel_size=2, padding=0),
#                       nn.Upsample(scale_factor=2, mode='nearest')]
#
#         return nn.Sequential(*model)
#
#     def forward(self, conv1):
#         # naming is according to the paper
#         conv2 = self.conv_block1(conv1)
#         concat1 = torch.cat([conv1, conv2], 1)
#
#         conv3 = self.conv_block2(concat1)
#         concat2 = torch.cat([conv2, conv3], 1)
#
#         conv4 = self.conv_block3(concat2)
#         concat3 = torch.cat([conv1,conv2,conv3,conv4],1)
#
#         return concat3
#
# class BReLU(nn.Module):
#     def __init__(self, up_thred, down_thred, up_value, down_value, inplace=False):
#         super(BReLU, self).__init__()
#         self.up_threshold = up_thred
#         self.down_threshold = down_thred
#         self.up_value = up_value
#         self.down_value = down_value
#         self.inplace = inplace
#
#     def forward(self, input):
#         temp = nn.functional.threshold(input, self.down_threshold, self.down_value, self.inplace)
#         return -nn.functional.threshold(-temp, -self.up_threshold, -self.up_value, self.inplace)
#
#     def __repr__(self):
#         inplace_str = ', inplace' if self.inplace else ''
#         return self.__class__.__name__ + ' (' \
#                 + str(self.up_threshold) \
#                 + str(self.down_threshold) \
#                 + ', ' + str(self.up_value) \
#                 + ', ' + str(self.down_value) \
#                 + inplace_str + ')'
#
# # Defines the generator that consists of Resnet blocks between a few
# # downsampling/upsampling operations.
# # Code and idea originally from Justin Johnson's architecture.
# # https://github.com/jcjohnson/fast-neural-style/
# class ResnetGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         self.gpu_ids = gpu_ids
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         n_downsampling = 2
#         for i in range(n_downsampling):
#             mult = 2**i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
#                                 stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]
#
#         mult = 2**n_downsampling
#         for i in range(n_blocks):
#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
#
#         for i in range(n_downsampling):
#             mult = 2**(n_downsampling - i)
# #            model += [nn.Upsample(scale_factor=2, mode='nearest'),
# #                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1, bias=use_bias),
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1, bias=use_bias),
#                       norm_layer(int(ngf * mult / 2)),
#                       nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#             print('Resnet_G并行加速')
#             return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#         else:
#             return self.model(input)


# class ResnetDepthGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', non_linearity=None, filtering=None, r=10, eps=1e-3):
#         assert(n_blocks >= 0)
#         super(ResnetDepthGenerator, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         self.gpu_ids = gpu_ids
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         self.filtering=filtering
#         self.non_linearity = non_linearity
#         if non_linearity is None:
#             last_act = nn.Tanh()
#         elif non_linearity == 'BReLU':
#             last_act = BReLU(0.95,0.05,0.95,0.05,True)
#         elif non_linearity == 'ReLU':
#             last_act = nn.ReLU(True)
#         elif non_linearity == 'sigmoid':
#             last_act = nn.Sigmoid()
#         elif non_linearity == 'linear':
#             last_act = None
#         else:
#             print(non_linearity)
#             raise NotImplementedError
#
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         n_downsampling = 2
#         for i in range(n_downsampling):
#             mult = 2**i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
#                                 stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]
#
#         mult = 2**n_downsampling
#         for i in range(n_blocks):
#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
#
#         for i in range(n_downsampling):
#             mult = 2**(n_downsampling - i)
# #            model += [nn.Upsample(scale_factor=2, mode='nearest'),
# #                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1, bias=use_bias),
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1, bias=use_bias),
#                       norm_layer(int(ngf * mult / 2)),
#                       nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         if last_act is not None:
#             model += [last_act]
#
#         self.model = nn.Sequential(*model)
#
#         if filtering is not None:
#             if filtering == 'max':
#                 self.last_layer = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
#             elif filtering == 'guided':
#                 self.last_layer = GuidedFilter(r=r, eps=eps)
#
#     def forward(self, input):
#         if self.filtering == 'guided':
#             # rgb2gray
#             guidance = 0.2989 * input[:,0,:,:] + 0.5870 * input[:,1,:,:] + 0.1140 * input[:,2,:,:]
#             # rescale to [0,1]
#             guidance = (guidance + 1) / 2
#             guidance = torch.unsqueeze(guidance, dim=1)
#
#         if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#             pre_filter = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#             if self.non_linearity is None:
#                 # rescale to [0,1]
#                 pre_filter = (pre_filter + 1) / 2
#
#             if self.filtering is not None:
#                 if self.filtering == 'guided':
#                     return pre_filter, self.last_layer(guidance, pre_filter)
#                 else:
#                     return pre_filter, nn.parallel.data_parallel(self.last_layer, pre_filter, self.gpu_ids)
#             else:
#                 return None, pre_filter
#         else:
#             pre_filter = self.model(input)
#             if self.non_linearity is None:
#                 # rescale to [0,1]
#                 pre_filter = (pre_filter + 1) / 2
#
#             if self.filtering is not None:
#                 if self.filtering == 'guided':
#                     return pre_filter, self.last_layer(guidance, pre_filter)
#                 else:
#                     return pre_filter, self.last_layer(pre_filter)
#             else:
#                 return None, pre_filter


# # Define a resnet block
# class ResnetBlock(nn.Module):#相邻两层
#     def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         super(ResnetBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias=use_bias)
#
#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         conv_block = []
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim),
#                        nn.ReLU(True)]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]
#
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim)]
#
#         return nn.Sequential(*conv_block)
#
#     def forward(self, x):
#         out = x + self.conv_block(x)#res是两组特征图元素级别的相加，特征图张数不变    skipconnect是特征图张数翻倍
#         return out


# # Defines the Unet generator.
# # |num_downs|: number of downsamplings in UNet. For example,
# # if |num_downs| == 7, image of size 128x128 will become of size 1x1
# # at the bottleneck
# class UnetGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64,
#                  norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], non_linearity=None, filtering=None, r=10, eps=1e-3):
#         super(UnetGenerator, self).__init__()
#         self.gpu_ids = gpu_ids
#         self.filtering=filtering
#         self.non_linearity = non_linearity
#         if non_linearity is None:
#             last_act = nn.Tanh()
#         elif non_linearity == 'BReLU':
#             last_act = BReLU(0.95,0.05,0.95,0.05,True)
#         elif non_linearity == 'ReLU':
#             last_act = nn.ReLU(True)
#         elif non_linearity == 'sigmoid':
#             last_act = nn.Sigmoid()
#         elif non_linearity == 'linear':
#             last_act = None
#         else:
#             print(non_linearity)
#             raise NotImplementedError
#
#         # currently support only input_nc == output_nc
# #        assert(input_nc == output_nc)
#
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
#         for i in range(num_downs - 5):
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, non_linearity=last_act)
#
#         self.model = unet_block
#
#         if filtering is not None:
#             if filtering == 'max':
#                 self.last_layer = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
#             elif filtering == 'guided':
#                 self.last_layer = GuidedFilter(r=r, eps=eps)
#
#     def forward(self, input):
#         if self.filtering == 'guided':
#             # rgb2gray
#             guidance = 0.2989 * input[:,0,:,:] + 0.5870 * input[:,1,:,:] + 0.1140 * input[:,2,:,:]
#             # rescale to [0,1]
#             guidance = (guidance + 1) / 2
#             guidance = torch.unsqueeze(guidance, dim=1)
#
#         #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#         if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
#             print('并行加速')
#             print(self.gpu_ids)
#             pre_filter = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#             if self.non_linearity is None:
#                 # rescale to [0,1]
#                 pre_filter = (pre_filter + 1) / 2
#
#             if self.filtering is not None:
#                 if self.filtering == 'guided':
#                     return pre_filter, self.last_layer(guidance, pre_filter)
#                 else:
#                     return pre_filter, nn.parallel.data_parallel(self.last_layer, pre_filter, self.gpu_ids)
#             else:
#                 return None, pre_filter
#         else:
#             pre_filter = self.model(input)
#             if self.non_linearity is None:
#                 # rescale to [0,1]
#                 pre_filter = (pre_filter + 1) / 2
#
#             if self.filtering is not None:
#                 if self.filtering == 'guided':
#                     return pre_filter, self.last_layer(guidance, pre_filter)
#                 else:
#                     return pre_filter, self.last_layer(pre_filter)
#             else:
#                 return None, pre_filter


# # Defines the submodule with skip connection.
# # X -------------------identity---------------------- X
# #   |-- downsampling -- |submodule| -- upsampling --|
# class UnetSkipConnectionBlock(nn.Module):#相隔多层  对称型
#     def __init__(self, outer_nc, inner_nc,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, non_linearity=None):
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         # modify it temporally for depth output
#         downconv = nn.Conv2d(3 if outermost else outer_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(outer_nc)
#
#         if outermost:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downconv]
# #            up = [uprelu, upconv, nn.Tanh()]
#             # modify it temporally for depth output
#             up = [uprelu, upconv, non_linearity]
#             model = down + [submodule] + up
#         elif innermost:
#             upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#             model = down + up
#         else:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]
#
#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         if self.outermost:
#             y = self.model(x)
#             print(y.size(), x.size())
#             return y
#         else:
#             y = self.model(x)
#             print (y.size(), x.size())
#             return torch.cat([y, x], 1)


# # Defines the PatchGAN discriminator with the specified arguments.
# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
#         super(NLayerDiscriminator, self).__init__()
#         self.gpu_ids = gpu_ids
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4
#         padw = int(np.ceil((kw-1)/2))
#         sequence = [
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                       kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
#
#         if use_sigmoid:
#             sequence += [nn.Sigmoid()]
#
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
#             return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#         else:
#             return self.model(input)
#
# # Defines the Multiscale-PatchGAN discriminator with the specified arguments.
# class MultiDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
#         super(MultiDiscriminator, self).__init__()
#         self.gpu_ids = gpu_ids
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         # cannot deal with use_sigmoid=True case at thie moment
#         assert(use_sigmoid == False)
#
#         kw = 4
#         padw = int(np.ceil((kw-1)/2))
#         scale1 = [
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, 3):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             scale1 += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         self.scale1 = nn.Sequential(*scale1)
#         scale1_output = []
#         scale1_output += [
#             nn.Conv2d(ndf * nf_mult, ndf * nf_mult,
#                       kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#         scale1_output += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]    # compress to 1 channel
#         self.scale1_output = nn.Sequential(*scale1_output)
#
#         scale2 = []
#         nf_mult = nf_mult
#         for n in range(3, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             scale2 += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         scale2 += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                       kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         scale2 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
#
#         if use_sigmoid:
#             scale2 += [nn.Sigmoid()]
#
#         self.scale2 = nn.Sequential(*scale2)
#
#     def forward(self, input):
#         if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
#             scale1 = nn.parallel.data_parallel(self.scale1, input, self.gpu_ids)
#             output1 = nn.parallel.data_parallel(self.scale1_output, scale1, self.gpu_ids)
#             output2 = nn.parallel.data_parallel(self.scale2, scale1, self.gpu_ids)
#         else:
#             scale1 = self.scale1(input)
#             output1 = self.scale1_output(scale1)
#             output2 = self.scale2(scale1)
#
#         return output1, output2
#
#
#
# #define GatedCoarsestNet
# class GatedCoarsestNetGenerator(nn.Module):
#     def __init__(self, input_nc=12, output_nc=1):
#         super(GatedCoarsestNetGenerator, self).__init__()
#         #encoder
#         #池化得到原图1/4大小
#         self.pool4 = nn.MaxPool2d(kernel_size=4,stride= 4, padding=0, dilation=0, return_indices=False, ceil_mode=False)
#         #e1
#         self.s1_e1conv1 = nn.Conv2d(in_channels=input_nc, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
#         self.relu_s1e1c1 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_e1conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1e1c2 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_e1conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1e1c3 = nn.PReLU(num_parameters=0, init=0.1)
#         #e2
#         self.s1_e2conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
#         self.relu_s1e2c1 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_e2conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1e2c1 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_e2conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1e2c3 = nn.PReLU(num_parameters=0, init=0.1)
#         #e3
#         self.s1_e3conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
#         self.relu_s1e3c1 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_e3conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1e3c2 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_e3conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1e3c3 = nn.PReLU(num_parameters=0, init=0.1)
#
#         #decoder
#         #d1
#         self.s1_d1conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
#         self.relu_s1d1c1 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_d1conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1d1c2 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_d1conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1d1c3 = nn.PReLU(num_parameters=0, init=0.1)
#         #d2
#         self.s1_d2conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
#         self.relu_s1d2c1 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_d2conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1d2c2 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_d2conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1d2c3 = nn.PReLU(num_parameters=0, init=0.1)
#         #d3
#         self.s1_d3conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
#         self.relu_s1d3c1 = nn.PReLU(num_parameters=0, init=0.1)
#         self.s1_d3conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1d3c2 = nn.PReLU(num_parameters=0, init=0.1)
#         #concat relu_s1e1c3,relu_s1e2c3,relu_s1e3c3,relu_s1d1c3,relu_s1d2c3,relu_s1d3c2,generate s1_concat_features
#         self.s1_d3conv3 = nn.Conv2d(in_channels=32*6, out_channels=3, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
#         self.relu_s1d3c3 = nn.PReLU(num_parameters=0, init=0.1)#得到三个confidence map
#         #分成三张图
#
#
#     def forward(self, input):
#         #测试有无中间变量的训练时间区别
#
#         coarsest_input = self.pool4(input)
#         [s1_haze, s1_whiteb, s1_contr, s1_gamma] = torch.chunk(coarsest_input, 4, dim=0)
#         #e1
#         s1_e1conv1 = self.s1_e1conv1(coarsest_input)
#         relu_s1e1c1 = self.relu_s1e1c1(s1_e1conv1)
#         s1_e1conv2 = self.s1_e1conv2(relu_s1e1c1)
#         relu_s1e1c2 = self.relu_s1e1c2(s1_e1conv2)
#         s1_e1conv3 = self.s1_e1conv3(relu_s1e1c2)
#         relu_s1e1c3 = self.relu_s1e1c3(s1_e1conv3)
#         #e2
#         s1_e2conv1 = self.s1_e2conv1(relu_s1e1c3)
#         relu_s1e2c1 = self.relu_s1e2c1(s1_e2conv1)
#         s1_e2conv2 = self.s1_e2conv2(relu_s1e2c1)
#         relu_s1e2c2 = self.relu_s1e2c2(s1_e2conv2)
#         s1_e2conv3 = self.s1_e2conv3(relu_s1e2c2)
#         relu_s1e2c3 = self.relu_s1e2c3(s1_e2conv3)
#         #e3
#         s1_e3conv1 = self.s1_e3conv1(relu_s1e2c3)
#         relu_s1e3c1 = self.relu_s1e3c1(s1_e3conv1)
#         s1_e3conv2 = self.s1_e3conv2(relu_s1e3c1)
#         relu_s1e3c2 = self.relu_s1e3c2(s1_e3conv2)
#         s1_e3conv3 = self.s1_e3conv3(relu_s1e3c2)
#         relu_s1e3c3 = self.relu_s1e3c3(s1_e3conv3)
#         #d1
#         s1_d1conv1 = self.s1_d1conv1(relu_s1e3c3)
#         relu_s1d1c1 = self.relu_s1d1c1(s1_d1conv1)
#         s1_d1conv2 = self.s1_d1conv2(relu_s1d1c1)
#         relu_s1d1c2 = self.relu_s1d1c2(s1_d1conv2)
#         s1_d1conv3 = self.s1_d1conv3(relu_s1d1c2)
#         relu_s1d1c3 = self.relu_s1d1c3(s1_d1conv3)
#         #d2
#         s1_d2conv1 = self.s1_d2conv1(relu_s1d1c3)
#         relu_s1d2c1 = self.relu_s1d2c1(s1_d2conv1)
#         s1_d2conv2 = self.s1_d2conv2(relu_s1d2c1)
#         relu_s1d2c2 = self.relu_s1d2c2(s1_d2conv2)
#         s1_d2conv3 = self.s1_d2conv3(relu_s1d2c2)
#         relu_s1d2c3 = self.relu_s1d2c3(s1_d2conv3)
#         #d3
#         s1_d3conv1 = self.s1_d3conv1(relu_s1d2c3)
#         relu_s1d3c1 = self.relu_s1d3c1(s1_d3conv1)
#         s1_d3conv2 = self.s1_d3conv2(relu_s1d3c1)
#         relu_s1d3c2 = self.relu_s1d3c2(s1_d3conv2)
#         #聚合 relu_s1e1c3,relu_s1e2c3,relu_s1e3c3,relu_s1d1c3,relu_s1d2c3,relu_s1d3c2,得到 s1_concat_features
#         print("relu_s1e1c3 size", relu_s1e1c3.size())
#         print("relu_s1e2c3 size", relu_s1e2c3.size())
#         print("relu_s1e3c3 size", relu_s1e3c3.size())
#         print("relu_s1d1c3 size", relu_s1d1c3.size())
#         print("relu_s1d2c3 size", relu_s1d2c3.size())
#         print("relu_s1d3c2 size", relu_s1d3c2.size())
#         s1_concat_features = torch.cat((relu_s1e1c3, relu_s1e2c3,relu_s1e3c3,relu_s1d1c3,relu_s1d2c3,relu_s1d3c2), dim=0)
#         #打印尺寸,合并的全部是relu后的结果
#         print("s1_concat_features size", s1_concat_features.size())
#         s1_d3conv3 = self.s1_d3conv3(s1_concat_features)
#         relu_s1d3c3 = self.relu_s1d3c3(s1_d3conv3)
#         #切分relu_s1d3c3得到三个衍生图各自的confidence map
#         [s1_w_whiteb,s1_w_contr,s1_w_gamma] = torch.chunk(relu_s1d3c3, 3, dim=0)
#         s1_w3_whiteb = torch.cat((s1_w_whiteb, s1_w_whiteb, s1_w_whiteb), dim=0)
#         s1_w3_contr = torch.cat((s1_w_contr, s1_w_contr, s1_w_contr), dim=0)
#         s1_w3_gamma = torch.cat((s1_w_gamma, s1_w_gamma, s1_w_gamma), dim=0)
#         s1_white_comp = s1_w3_whiteb * s1_whiteb
#         s1_contr_comp = s1_w3_contr * s1_contr
#         s1_gamma_comp = s1_w3_gamma * s1_gamma
#         #print(s1_w3_whiteb)
#         #print(s1_whiteb)
#         #print(s1_white_comp)
#         #print(s1_w3_contr)
#         #print(s1_contr)
#         #print(s1_contr_comp)
#         #print(s1_w3_gamma)
#         #print(s1_gamma)
#         #print(s1_gamma_comp)
#         s1_dehazed = s1_white_comp + s1_contr_comp + s1_gamma_comp
#         #print(s1_dehazed)
#         return s1_dehazed
#
#

# class IQANetGenerator(nn.Module):
#     def __init__(self, weights_path, device):
#         super(IQANetGenerator, self).__init__()
#
#         weights = dd.io.load(weights_path)
#         stages = {'conv1': [], 'conv2': [], 'conv3': [], 'conv4': [], 'subtask1_fc1': [], 'subtask1_fc2': [], 'subtask2_fc1': [], 'subtask2_fc2': []}
#
#         for stage, module in stages.items():
#             if 'conv' in stage:
#                 stride = 2
#                 padding = 2
#                 if 'conv4' in stage:
#                     stride = 1
#                     padding = 0
#
#                 layer = nn.Conv2d(in_channels=weights[stage + '_weights'].shape[1], out_channels=weights[stage + '_weights'].shape[0], kernel_size=weights[stage + '_weights'].shape[2],
#                                   stride=stride, padding=padding, dilation=1, groups=1, bias=True)
#                 layer.weight.data = weights[stage + '_weights']
#                 layer.bias.data = weights[stage + '_biases']
#                 module.append(layer)
#                 layer = GDN(weights[stage + '_weights'].shape[0], device)
#                 layer._parameters['beta'] = weights[stage + '_gdn_x_reparam' + '_beta']
#                 layer._parameters['gamma'] = weights[stage + '_gdn_x_reparam' + '_gamma']
#                 module.append(layer)
#                 module.append(nn.MaxPool2d(kernel_size=2,stride=2))
#             else:
#                 layer = nn.Linear(weights[stage + '_weights'].shape[1], weights[stage + '_weights'].shape[0])
#                 layer.weight.data = weights[stage + '_weights'].squeeze()
#                 layer.bias.data = weights[stage + '_biases']
#                 module.append(layer)
#                 if 'fc1' in stage:
#                     layer = GDN(weights[stage + '_weights'].shape[0], device)
#                     layer._parameters['beta'] = weights[stage + '_gdn_x_reparam' + '_beta']
#                     layer._parameters['gamma'] = weights[stage + '_gdn_x_reparam' + '_gamma']
#                     module.append(layer)
#                 if 'subtask1_fc2' in stage:
#                     module.append(nn.Softmax2d())
#             self.add_module(stage, nn.Sequential(*module))
#     def forward(self, input, assess = False):
#         conv1 = self.conv1(input)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         conv4 = self.conv4(conv3).squeeze()
#
#         s1fc1 = self.subtask1_fc1._modules['0'](conv4)
#         s1fc1 = self.subtask1_fc1._modules['1'](s1fc1.unsqueeze(2).unsqueeze(3))
#         s1fc2 = self.subtask1_fc2._modules['0'](s1fc1.squeeze())
#         s1fc2 = self.subtask1_fc2._modules['1'](s1fc2.unsqueeze(2).unsqueeze(3))
#
#         s2fc1 = self.subtask2_fc1._modules['0'](conv4)
#         s2fc1 = self.subtask2_fc1._modules['1'](s2fc1.unsqueeze(2).unsqueeze(3))
#         s2fc2 = self.subtask2_fc2(s2fc1.squeeze())
#         if assess is True:
#             dist = s1fc2.squeeze()
#             score =  dist * s2fc2
#             return dist.cpu().numpy(),score.cpu().numpy()#s1fc2 is prob
#         else:
#             return OrderedDict([('conv1', conv1),('conv2',conv1),('conv3',conv3),('s1fc2',s1fc2),('s2fc2',s2fc2)])
#
#     def __generate_patches__(self, img, input_size=256, type=np.float32):
#         img_shape = img.shape
#         img = img.astype(dtype=type)
#         if len(img_shape) == 2:
#             H, W = img_shape
#             ch = 1
#         else:
#             H, W, ch = img_shape
#         if ch == 1:
#             img = np.asarray([img, ] * 3, dtype=img.dtype)
#             ch = 3
#
#         stride = int(input_size / 2)
#         hIdxMax = H - input_size
#         wIdxMax = W - input_size
#
#         hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
#         if H - input_size != hIdx[-1]:
#             hIdx.append(H - input_size)
#         wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
#         if W - input_size != wIdx[-1]:
#             wIdx.append(W - input_size)
#         patches = [img[hId:hId + input_size, wId:wId + input_size, :]
#                    for hId in hIdx
#                    for wId in wIdx]
#         patches = np.asarray(patches).transpose((0, 3, 1, 2)).astype(np.float)
#
#         return patches
#
#     def predict(self,img_names, device, quiet=True):
#         if not isinstance(img_names, str) and not isinstance(img_names, list):
#             raise ValueError('The arg is neither an image name nor a list of image names')
#         if isinstance(img_names, str):
#             img_names = [img_names, ]
#
#         p_labels = []
#         p_scores = []
#         for file_name in img_names:
#             try:
#                 img = misc.imread(file_name)
#             except Exception:
#                 raise IOError('Fail to load image: %s' % file_name)
#
#             h,w,c = img.shape
#             if h<256 or w<256:
#                 scale = max(256.0/h,256.0/w)
#                 img = np.resize(img,(int(h*scale),int(w*scale),c))
#
#
#             patches = self.__generate_patches__(img, input_size=256)
#             patches = torch.from_numpy(patches).float().to(device)
#             patch_probs, patch_qs = self.forward(patches,assess=True)
#
#             patch_types = [p.argmax() for p in patch_probs]
#             img_type, _ = stats.mode(patch_types)
#
#             p_labels.append(img_type[0] + 1)
#             p_scores.append(np.mean(patch_qs))
#             if not quiet:
#                 print("%s: %.4f" % (file_name, p_scores[-1]))
#         return p_labels, p_scores

#define GatedNet
class GatedNetGenerator(nn.Module):
    def __init__(self, input_nc=12):
        super(GatedNetGenerator, self).__init__()

        stages = {'encoder1': [], 'encoder2': [], 'encoder3': [],'decoder1': [],'decoder2': [],'decoder3': []}

        for stage, layers in stages.items():
            if 'decoder3' in stage:
                l=2
            else:
                l=3
            for i in range(l):
                in_ch = 32
                out_ch = 32
                kernel = 3
                dilation = 1
                padding = 1
                if 'encoder1' in stage and i == 0:
                    in_ch = input_nc
                    kernel = 7

                if 'encoder1' in stage and i==0:
                    padding = 9
                elif ('encoder2' in stage and i==0) or ('encoder1' in stage and i!=0):
                    padding = 3

                if 'encoder1' in stage or ('encoder2' in stage and i==0):
                    dilation = 3

                if 'decoder' in stage and i == 0 :
                    layers.append(nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel,
                                                       dilation=dilation, padding=(padding, padding)))
                else:
                    layers.append(
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, dilation=dilation,
                                  padding=(padding, padding)))
                layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
            self.add_module(stage, nn.Sequential(*layers))

            decoder3_concat = []
            decoder3_concat.append(nn.Conv2d(in_channels=32 * 6, out_channels=3, kernel_size=3, dilation=1, stride=1, padding=(1, 1), groups=1,
                      bias=True))
            decoder3_concat.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
            self.add_module('decoder3_concat', nn.Sequential(*decoder3_concat))



        # #encoder
        # ##------------------------------------------------------Scale 3-----------------------------------------------------------##
        # #e1
        # self.s3_e1conv1 = nn.Conv2d(in_channels=input_nc, out_channels=32, kernel_size=7, dilation=3, stride=1,padding=(9,9),groups=1, bias=True)
        # self.relu_s3e1c1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_e1conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=3, stride=1,padding=(3, 3), groups=1, bias=True)
        # self.relu_s3e1c2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_e1conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=3, stride=1,padding=(3, 3), groups=1, bias=True)
        # self.relu_s3e1c3 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # #e2
        # self.s3_e2conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,dilation=3, stride=1,padding=(3,3),groups=1, bias=True)
        # self.relu_s3e2c1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_e2conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3e2c2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_e2conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3e2c3 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # #e3
        # self.s3_e3conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
        # self.relu_s3e3c1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_e3conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3e3c2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_e3conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3e3c3 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # #decoder
        # #d1
        # self.s3_d1conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
        # self.relu_s3d1c1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_d1conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3d1c2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_d1conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3d1c3 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # #d2
        # self.s3_d2conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
        # self.relu_s3d2c1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_d2conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3d2c2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_d2conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3d2c3 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # #d3
        # self.s3_d3conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3,dilation=1, stride=1,padding=(1,1),groups=1, bias=True)
        # self.relu_s3d3c1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # self.s3_d3conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3d3c2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        # #concat relu_s3e1c3,relu_s3e2c3,relu_s3e3c3,relu_s3d1c3,relu_s3d2c3,relu_s3d3c2,generate s3_concat_features
        # self.s3_d3conv3 = nn.Conv2d(in_channels=32*6, out_channels=3, kernel_size=3, dilation=1, stride=1,padding=(1, 1), groups=1, bias=True)
        # self.relu_s3d3c3 = nn.LeakyReLU(negative_slope=0.1, inplace=False)#得到三个confidence map

    def forward(self, input):
        [img, ch, lc, lg] = torch.chunk(input, 4, dim=1)
        del img

        feature_e1 = self.encoder1(input)
        feature_e2 = self.encoder2(feature_e1)
        feature_e3 = self.encoder3(feature_e2)
        feature_d1 = self.decoder1(feature_e3)
        feature_d2 = self.decoder2(feature_d1)
        feature_d3 = self.decoder3(feature_d2)
        feature_d3 = torch.cat((feature_e1,feature_e2,feature_e3,feature_d1,feature_d2,feature_d3), dim=1)
        confi_maps = self.decoder3_concat(feature_d3)

        [ch_map, lc_map, lg_map] = torch.chunk(confi_maps, 3, dim=1)
        ch_map = torch.cat((ch_map, ch_map, ch_map), dim=1)
        lc_map = torch.cat((lc_map, lc_map, lc_map), dim=1)
        lg_map = torch.cat((lg_map, lg_map, lg_map), dim=1)

        ch = ch_map * ch
        lc = lc_map * lc
        lg = lg_map * lg
        enhanced = ch  + lc + lg

        return OrderedDict([('en', enhanced), ('ch_map', ch_map),
                ('lc_map', lc_map), ('lg_map', lg_map)])

        # #-----------------------------S3---------------------------#
        # out = self.s3_e1conv1(input)
        # out = self.relu_s3e1c1(out)
        # out = self.s3_e1conv2(out)
        # out = self.relu_s3e1c2(out)
        # out = self.s3_e1conv3(out)
        # relu_s3e1c3 = self.relu_s3e1c3(out)
        # #e2
        # out = self.s3_e2conv1(relu_s3e1c3)
        # out = self.relu_s3e2c1(out)
        # out = self.s3_e2conv2(out)
        # out = self.relu_s3e2c2(out)
        # out = self.s3_e2conv3(out)
        # relu_s3e2c3 = self.relu_s3e2c3(out)
        # #e3
        # out = self.s3_e3conv1(relu_s3e2c3)
        # out = self.relu_s3e3c1(out)
        # out = self.s3_e3conv2(out)
        # out = self.relu_s3e3c2(out)
        # out = self.s3_e3conv3(out)
        # relu_s3e3c3 = self.relu_s3e3c3(out)
        # #d1
        # out = self.s3_d1conv1(relu_s3e3c3)
        # out = self.relu_s3d1c1(out)
        # out = self.s3_d1conv2(out)
        # out = self.relu_s3d1c2(out)
        # out = self.s3_d1conv3(out)
        # relu_s3d1c3 = self.relu_s3d1c3(out)
        # #d2
        # out = self.s3_d2conv1(relu_s3d1c3)
        # out = self.relu_s3d2c1(out)
        # out = self.s3_d2conv2(out)
        # out = self.relu_s3d2c2(out)
        # out = self.s3_d2conv3(out)
        # relu_s3d2c3 = self.relu_s3d2c3(out)
        # #d3
        # out = self.s3_d3conv1(relu_s3d2c3)
        # out = self.relu_s3d3c1(out)
        # out = self.s3_d3conv2(out)
        # relu_s3d3c2 = self.relu_s3d3c2(out)
        # #聚合 relu_s1e1c3,relu_s1e2c3,relu_s1e3c3,relu_s1d1c3,relu_s1d2c3,relu_s1d3c2,得到 s1_concat_features
        # out = torch.cat((relu_s3e1c3, relu_s3e2c3,relu_s3e3c3,relu_s3d1c3,relu_s3d2c3,relu_s3d3c2), dim=1)
        #
        # del relu_s3e1c3, relu_s3e2c3,relu_s3e3c3,relu_s3d1c3,relu_s3d2c3,relu_s3d3c2
        #
        # #打印尺寸,合并的全部是relu后的结果
        # # print("s1_concat_features size", s1_concat_features.size())
        # out = self.s3_d3conv3(out)
        # out = self.relu_s3d3c3(out)
        # #切分relu_s1d3c3得到三个衍生图各自的confidence map
        # [s3_w_wb, s3_w_lc, s3_w_gm] = torch.chunk(out, 3, dim=1)
        # s3_w3_wb = torch.cat((s3_w_wb, s3_w_wb, s3_w_wb), dim=1)
        # s3_w3_lc = torch.cat((s3_w_lc, s3_w_lc, s3_w_lc), dim=1)
        # s3_w3_gm = torch.cat((s3_w_gm, s3_w_gm, s3_w_gm), dim=1)
        #
        # s3_wb_comp = s3_w3_wb * s3_wb
        # s3_lc_comp = s3_w3_lc * s3_lc
        # s3_gm_comp = s3_w3_gm * s3_gm
        # s3_enhanced = s3_wb_comp  + s3_lc_comp + s3_gm_comp
        #
        # return OrderedDict([('s3en', s3_enhanced), ('w3wb', s3_w3_wb),
        #         ('w3lc', s3_w3_lc), ('w3gm', s3_w3_gm)])



# # Define RefineNet
# class RefineNetGenerator(nn.Module):
#     def __init__(self, input_nc=4, output_nc=1):
#         super(RefineNetGenerator, self).__init__()
#         self.conv1 = nn.Conv2d(input_nc, 5, kernel_size=7, padding=(3,3), bias=True)
#         self.relu1 = nn.ReLU(True)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0)
#         self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.bn1 = nn.BatchNorm2d(5)
#
#         self.conv2 = nn.Conv2d(5, 5, kernel_size=5, padding=(2,2), bias=True)
#         self.relu2 = nn.ReLU(True)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0)
#         self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.bn2 = nn.BatchNorm2d(5)
#
#         self.conv3 = nn.Conv2d(5, 10, kernel_size=3, padding=(1,1), bias=True)
#         self.relu3 = nn.ReLU(True)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, padding=0)
#         self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.ln = nn.Linear(10, 1)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, input):
#         #print('layer1')
#         out = self.conv1(input)
#         #print(out.size())
#         out = self.relu1(out)
#         #print(out.size())
#         out = self.pool1(out)
#         #print(out.size())
#         out = self.up1(out)
#         #print(out.size())
#         out = self.bn1(out)
#         #print(out.size())
#
#         #print('layer2')
#
#         out = self.conv2(out)
#         # print(out.size())
#         out = self.relu2(out)
#         # print(out.size())
#         out = self.pool2(out)
#         # print(out.size())
#         out = self.up2(out)
#         #print(out.size())
#         out = self.bn2(out)
#         # print(out.size())
#
#         #print('layer3')
#         out = self.conv3(out)
#         #print(out.size())
#         out = self.relu3(out)
#         #print(out.size())
#         out = self.pool3(out)
#         #print(out.size())
#         out = self.up3(out)
#         #print(out.size())
#
#         out = out.transpose(1,3)
#         #print(out.size())
#
#
#         out = self.ln(out)
#         #print(out.size())
#
#         out = out.transpose(1, 3)
#         #print(out.size())
#
#         out = self.sig(out)
#         #print(out.size())
#
#         return out