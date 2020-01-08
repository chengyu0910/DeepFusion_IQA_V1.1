#Eval the dark dataset image scores to make sure that the MEON is adaptived with these data, i.e
#whether the label images quality is better than sythsized dark images quality(assessed by MEON).

from models.networks import IQANetGenerator
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt

gpu_id_map_table = [5, 2, 4, 0, 1, 3, 6, 5]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.backends.cudnn.enabled = False

label_dir = '/data0/data_cy/DeepFusion_dataset_agml/label/'
dark_train_dir = '/data0/data_cy/DeepFusion_dataset_agml/data/trainset/'
dark_test_dir = '/data0/data_cy/DeepFusion_dataset_agml/data/testset/'
label_names_all = sorted(make_dataset(label_dir))
sample_ids = sorted(random.sample(range(len(label_names_all)),200))
# label_names = []
# for id in sample_ids:
#     label_names.append(label_names_all[id])
label_names = label_names_all

dark_train_names = os.listdir(dark_train_dir)
dark_test_names = os.listdir(dark_test_dir)
synth_names = []
for label_name in label_names:
    sample_names = []
    for train_name in dark_train_names:
        if label_name in train_name:
            sample_names.append(dark_train_dir+train_name)
    if len(sample_names) is 0:
        for test_name in dark_test_names:
            if label_name in test_name:
                sample_names.append(dark_test_dir + test_name)
    id = random.randint(0,6)
    synth_names.append(sample_names[id])

label_names = [label_dir+name for name in label_names]

device  = torch.device('cuda')
weights_path = './MEON-0.h5'
MEON = IQANetGenerator(weights_path,device)
MEON = MEON.to(device)

with torch.no_grad():
    dis_label, scr_label = MEON.predict(label_names,device)
    dis_synth, scr_synth = MEON.predict(synth_names,device)

err = 0
for i in range(len(scr_label)):
    if scr_label[i]>scr_synth[i]:
        err += 1
print(err)

x=np.arange(len(label_names))
plt.figure()
plt.plot(x, scr_label, 'r-', x, scr_synth, 'g-')


