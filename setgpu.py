import os
import torch

def setgup(id):
    gpu_id_map_table = [5, 2, 4, 0, 1, 3, 6, 5]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)
    torch.backends.cudnn.enabled = False