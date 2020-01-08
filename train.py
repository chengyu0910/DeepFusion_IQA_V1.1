#coding=utf-8
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualize_custom import Visualizer_custom
from PIL import Image
from collections import OrderedDict
import datetime
from util import util
import torch
import os
import numpy as np

def learning_rate_updata(epoch, model, opt, restore=False):
    nodes = eval(opt.lr_decay_nodes)
    for node in nodes:
        if (restore is False and epoch==node) or (restore is True and epoch>=node):
            if opt.lr_decay_mode is 'linear':
                model.update_learning_rate(model.get_learning_rate()*opt.lr_decay_param)
            elif opt.lr_decay_mode is 'exp':
                None
            else:
                None
    return

if __name__ == '__main__':
    gpu_id_map_table = [5,2,4,0,1,3,6,5]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    torch.backends.cudnn.enabled = False

    opt = TrainOptions().parse()  # 得到网络模型的options
    print('获取输入参数')
    data_loader = CreateDataLoader(opt)
    print('创建样本迭代器')
    # dataset = data_loader.load_data()
    # print('创建样本集')
    # dataset_size = len(dataset)
    # print('获取样本集大小')
    # print('#training images = %d' % dataset_size)

    model = create_model(opt)  # 根据options创建模型
    print('创建网络模型')
    visualizer = Visualizer_custom(netname=opt.name, snapdir=opt.checkpoints_dir, restore_train=opt.restore_train)  # 可视化模型
    print('可视化模型')

    epochtime = 0
    total_itertime = 0

    if opt.restore_train is True:
        epoch = opt.which_epoch
        learning_rate_updata(epoch,model,opt,restore=True)#set lr corresponding to restored epoch
    else:
        epoch = 0

    while epoch < opt.max_epoch:
        epoch += 1
        epoch_start_time = time.time()
        # reinitialize data_load at each epoch and shuffle
        data_loader.initialize(opt)
        trainset = data_loader.load_traindata()
        print('开始第 %d 次迭代，重新打乱数据' % (epoch))

        for i, data in enumerate(trainset):
            iter_start_time = time.time()
            total_iter = (epoch-1)*len(trainset)+i+1

            model.set_input(data)
            model.optimize_parameters()

            if opt.display is True and total_iter % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals())

            if total_iter % opt.print_freq == 0:
                loss = model.get_current_errors()
                loss_train = dict([])
                for k,v in loss.items():
                    loss_train[k+'_train'] = v
                visualizer.print_current_errors(epoch, epochtime, total_iter, loss_train, total_itertime/total_iter, resttime=max((len(trainset)*opt.max_epoch-total_iter)*total_itertime/total_iter, (opt.max_epoch-epoch)*epochtime))
                # if total_iter % opt.log_freq == 0:
                #     visualizer.log_current_errors()
                # if total_iter % opt.plot_freq == 0:
                #     visualizer.plot_current_errors(loss_train, opt.plot_freq,opt.which_epoch)

            total_itertime += time.time() - iter_start_time


        #test
        loss = dict([])
        testset = data_loader.load_testdata()
        for j, testdata in enumerate(testset):
            print('第 %d 次测试iter %d '%(epoch,j+1))
            model.set_input(testdata)
            model.test()
            loss_batch = model.get_current_errors()

            if len(loss.keys()) is 0:
                for k, v in loss_batch.items():
                    loss[k] = v
            else:
                for k, v in loss_batch.items():
                    loss[k] += v

        loss_test = dict([])
        for k,v in loss.items():
            loss_test[k+'_test'] = v/len(testset)

        visualizer.plot_current_errors(loss_test, 1,opt.which_epoch)


        #upadate lr or not
        learning_rate_updata(epoch,model,opt)
        #checkpoint
        if epoch % opt.checkpoints_freq == 0:
            print('saving the latest model epoch:%d'%(epoch))
            model.save(epoch)
            model.save('latest')

        epochtime = time.time() - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' %(epoch, opt.max_epoch, epochtime))
