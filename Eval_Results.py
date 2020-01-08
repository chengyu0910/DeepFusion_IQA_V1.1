#Eval the results of our method and other methods.

from models.networks import IQANetGenerator
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import os
import cv2
import random
import matplotlib.pyplot as plt
from evaluation import niqe,ssim,psnr
import TrancatedIQA



def noref_eval(imgs_dir):
    niqes = []
    imgs_names = os.listdir(imgs_dir)
    for name in imgs_names:
        img = cv2.imread(imgs_dir + name)
        niqes.append(niqe.niqe(img))
    return niqes

def fullref_eval(imgs_dir, refs_dir):
    psnrs = []
    ssims = []
    imgs_names = sorted(os.listdir(imgs_dir))
    for name in imgs_names:
        ref_name = name#TODO:exact ref name in img name
        img = cv2.imread(imgs_dir + name)
        ref = cv2.imread(refs_dir + ref_name)
        psnrs.append(psnr.psnr(img, ref))
        ssims.append(ssim.ssim(img, ref))
    return psnrs,ssims



if __name__ == '__main__':
    results_dir = '/home/cy/Project/DeepFusion_Experiments/DeepFusion_IQA_V1.0/imgs/results_Trancated_real_410/'
    ref_dir = ''
    results = os.listdir(results_dir)
    mode = 'noref'#noref or fullref
    metrics = {'niqe': [], 'cnn': [], 'psnr': [], 'ssim': []}
    niqe_mean = 0
    ssim_mean = 0
    psnr_mean = 0
    cnn_mean = 0

    gpu_id_map_table = [5, 2, 4, 0, 1, 3, 6, 5]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda')

    if mode is 'noref':
        matfile = "/home/cy/Project/DeepFusion_Experiments/DeepFusion_IQA_V1.0/Trancated.mat"
        IQANet = TrancatedIQA.IQANet_trancated(matfile).to(device)

    for i,result in enumerate(results):
        print('processing %d/%d'%(i,len(results)))
        img = cv2.imread(results_dir + result)

        if mode is 'noref':
            niqe_score = niqe.niqe(img)
            metrics['niqe'].append({result: niqe_score})
            niqe_mean += niqe_score

            img_cnn = torch.from_numpy(img).to(device).permute(-1, 0, 1).unsqueeze(0).float() / 255.0#TODO: make sure that img dims&range&dtype is matched to cnn
            cnn_score = IQANet(img)
            metrics['cnn'].append({result: cnn_score})
            cnn_mean += niqe_score

        elif mode is 'fullref':
            ref = cv2.imread(ref_dir + result)#TODO:name sparse
            psnr_score = psnr.psnr(img, ref)
            ssim_score = ssim.ssim(img, ref)
            metrics['psnr'].append({result: psnr_score})
            metrics['ssim'].append({result: ssim_score})
            ssim_mean += ssim_score
            psnr_mean += psnr_score

        metrics['niqe'].append({'niqe_mean':niqe_mean/len(results)})
        metrics['cnn'].append({'cnn_mean': cnn_mean/len(results)})
        metrics['psnr'].append({'psnr_mean':psnr_mean/len(results)})
        metrics['ssim'].append({'ssim_mean':ssim_mean/len(results)})

    print(metrics)
    #evals[{'niqe': [], 'psnr': [], 'ssim': []},{'niqe': [], 'psnr': [], 'ssim': []},...]
    #plot historgram
