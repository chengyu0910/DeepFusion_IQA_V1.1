import numpy as np
import torch
import os
from torch.autograd import Variable
import util.util as util
import torch.nn as nn
from collections import OrderedDict
import torchvision.transforms as transforms


from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib
import numpy as np
from PIL import Image
from tools.dot import make_dot
from models.networks import GatedNetGenerator

from models.gatednet import GatedNetModel


import cv2
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
plt.ion()

imgsrc = cv2.imread('./inputs/01.bmp') #直接读为灰度图像

r, c, h = imgsrc.shape

img = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2HSV)

im = np.zeros(img.shape)
im[:,:,0] = img[:,:,0]
im[:,:,1] = img[:,:,1]
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
im[:,:,2] = clahe.apply(img[:,:,2])

im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_HSV2BGR)
# for k in range(h):
# 	temp = img[:,:,k]
# 	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
# 	im[:,:,k] = clahe.apply(temp)


im = im.astype(np.uint8)
cv2.imshow('clahe', im)
cv2.imshow('imgsrc', imgsrc)
a=1
# clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10,10))
# cl1 = clahe.apply(img)


# pic01=plt.plot(img)
# plt.show()                  #是plt.show()而不是pic01.show()

# plt.subplot(121),plt.imshow(img)
# plt.subplot(122),plt.imshow(cl1,'gray')





# model = GatedNetGenerator().cuda(0)
# x = Variable(torch.randn(10,12,128,128))#change 12 to the channel number of network input
# x = x.cuda(0)
# y = model.forward(x)
# g = make_dot(y['s3en'])
# g.view()

# A = Image.open('C:\\Users\\chengyu\\Desktop\\testimg\\epoch005_label.png').convert('RGB')
# B = Image.open('C:\\Users\\chengyu\\Desktop\\testimg\\epoch005_prediction.png').convert('RGB')
# img2tensor = transforms.ToTensor()
# loss = nn.MSELoss()
# A = Variable(img2tensor(A))
# B = Variable(img2tensor(B))
#
# A = torch.stack((A,A,A),dim = 0)
# B = torch.stack((A,B,B),dim = 0)
#
# L = loss(A,B)
# print('LOSS:',L)

# a = torch.tensor((1,2,3,4,5,6,7,8,9));
# a = a.view(3,3);
# b = torch.tensor((3,2,5,4,1,6,8,3,4));
# b = b.view(3,3);
# a = Variable(a)
# b = Variable(b)
# loss = torch.nn.MSELoss();
# L = loss(a,b);



# viz = Visdom()
#
# assert viz.check_connection()
# viz.close()
# img = np.random.rand(512, 256)
# print(type(img))
# #单张
# X = np.array([0, 1])
# Y = np.array([0, 1])
# print(type(X))
# print(type(Y))
# print(X)
# print(Y)
# win=viz.line(
#     X=X,
#     Y=Y,
#     opts=dict(
#         xtickmin=-2,
#         xtickmax=2,
#         xtickstep=1,
#         ytickmin=-1,
#         ytickmax=5,
#         ytickstep=1,
#         markersymbol='dot',
#         markersize=5,
#     ),
#     name="1"
# )
#
#
#
# viz.image(
#     img,
#     opts=dict(title='Random!', caption='How random.'),
#     win = 1
# )
# b = 0.009
# A = OrderedDict([('Refine_loss', b)])


#
# import numpy as np
# from PIL import Image
#
#
# img = Image.open('M:/dataset/ITS/val/trans/10001_01.png').convert('L')
# #img.show()
# print(img.size)
# print(type(img))
#
#
# path='ss/caf/'
# name = 1000
#
# print(path + "%i"%name)
# x=np.random.rand(500,500,10)
# x=x*255
# y=Image.fromarray(x[:,:,1])
# y.show()


# x=np.array([[1,2,3],[9,8,7],[6,5,4]])
# print(x)
# print(x.size)
# print(x.ndim)
# print(x.shape)
#
# y=np.squeeze(x)
# print(y)
# print(y.size)
# print(y.ndim)
# print(y.shape)

# img=torch.randn(5, 5)
#
# for i in range(4,-1,-1):
#     for j in range(4, -1, -1):
#         img[i,j]=(i*5+j)/10
#
# img = Variable(img)
# print(img)
# print(img.data)
# #a=img.data[0].cpu().float().numpy()
# a= util.tensor2im(img.data)
# print(a)
# print(type(a))
# print(a.size)
#
#
#
# ls = OrderedDict([('label', a), ('test', b)])
# print(type(ls))
# for label, now in ls.items():
#     print(label)
#     print(now)
#
# m = nn.Linear(20,30)
# input = Variable(torch.randn(128, 20, 100, 55))
# input = input.transpose(1,3);
# print(input.size())
#
# output = m(input)
# print(output.size())
# output = output.transpose(1,3)
# print(output.size())