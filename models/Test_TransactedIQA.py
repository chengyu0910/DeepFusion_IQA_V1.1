import torch
import TrancatedIQA
from PIL import Image
import numpy as np
import os

gpu_id_map_table = [5, 2, 4, 0, 1, 3, 6, 5]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.backends.cudnn.enabled = False
device = torch.device('cuda')


imgs = ['00003.bmp','00009.bmp','00012.bmp','00014.bmp','00019.bmp']
img_dir= '/data0/data_cy/DeepFusion_dataset_agml/label/'


matfile = "/home/cy/Project/DeepFusion_Experiments/DeepFusion_IQA_V1.0/Trancated.mat"
IQANet = TrancatedIQA.IQANet_trancated(matfile).to(device)

for name in imgs:
    img = np.array(Image.open(img_dir+name).convert('RGB'), dtype=np.float)
    img = torch.from_numpy(img).to(device).permute(-1, 0, 1).unsqueeze(0).float()/255.0
    score = IQANet(img)
    print(score['score'])