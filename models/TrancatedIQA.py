
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch
from models.BCNN import BCNN
#matlab文件名


class IQANet_trancated(nn.Module):
    def __init__(self, matfile):
        super(IQANet_trancated, self).__init__()
        # matfile = r"C:\Users\chengyu\Desktop\IQAloss\Hu\matlab_code\net.mat"
        dict = sio.loadmat(matfile)
        netdict = dict['net'].squeeze()
        num_layers = netdict.shape[0]
        stride =  [1,2,1,2,1,2,1,1,2]
        net_convs = []
        net_fc = []
        for i in range(int(num_layers/2)):
            layer_name = netdict[i*2][0][0]
            if 'regress' not in layer_name and 'conv' in layer_name:
                if i is 0:
                    in_chs = 3
                else:
                    in_chs = netdict[(i-1)*2][1].shape[-1]
                out_chs = netdict[i*2][1].shape[-1]
                conv = torch.nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=(3,3), stride=(stride[i],stride[i]),padding=(1,1))
                conv.weight.data = torch.from_numpy(netdict[i*2][1]).permute(3,2,0,1).float()
                conv.bias.data = torch.from_numpy(netdict[i*2+1][1]).squeeze().float()
                net_convs.append(conv)
                net_convs.append(torch.nn.ReLU())
            elif 'regress' in layer_name:
                fc = torch.nn.Linear(netdict[i*2][1].shape[-1],1)
                fc.weight.data = torch.from_numpy(netdict[i*2][1]).squeeze(0).float()
                fc.bias.data = torch.from_numpy(netdict[i*2+1][1]).squeeze().float()
                net_fc.append(fc)
        self.add_module('net_convs', nn.Sequential(*net_convs))
        self.add_module('net_fc', nn.Sequential(*net_fc))
        # self.net_convs = torch.nn.Sequential(*net_convs)
        # self.net_fc = torch.nn.Sequential(*net_fc)
        self.net_bilinear_pool = BCNN()

    def forward(self, input):#  Attention:input is in range (0,255)
        input = input*255
        input[:, 0, :, :] = input[:, 0, :, :] - 123.8181
        input[:, 1, :, :] = input[:, 1, :, :] - 119.8395
        input[:, 2, :, :] = input[:, 2, :, :] - 114.6756

        nodes_convs = [3,7,11,17]#net nodes for convsblock
        nodes_convs_name=['conv1','conv2','conv3','conv4']
        feat_and_score = dict([])
        cnt = 0
        for i in range(len(self.net_convs._modules)):
            if i is 0:
                feat = self.net_convs._modules[str(i)](input)
            else:
                feat = self.net_convs._modules[str(i)](feat)
            if i in nodes_convs:
                feat_and_score = dict(feat_and_score,**{nodes_convs_name[cnt]:feat})
                cnt += 1

        feat = self.net_bilinear_pool(feat)
        score = self.net_fc(feat)
        feat_and_score = dict(feat_and_score, **{'score':score})

        return feat_and_score







