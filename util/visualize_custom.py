#coding=utf-8
import numpy as np
import time
import os
import visdom
import datetime

class Visualizer_custom():
    def __init__(self, netname, snapdir, display_port=8097,restore_train=False):
        self.name = netname
        print('打开visdom')
        self.vis = visdom.Visdom(port = display_port,env=self.name)
        assert self.vis.check_connection()
        self.vis.close()

        self.log_name = os.path.join(snapdir, netname, 'loss_log.txt')
        self.log_freq = 50
        self.log_cnt = 0
        if restore_train is True:
            log_mode = 'a'
        else:
            log_mode = 'w'
        with open(self.log_name, log_mode) as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        self.plot_data = {}

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals):#visuals 是个字典['name1':img1,'name2':img2...] img是三维numpy格式数据[c,h,w]
        idx = 1
        imgs = []
        labels = ' '
        for label, img in visuals.items():
            labels += label + '/'
            img_01 = img
            img_01[img_01>1]=1
            img_01[img_01<0]=0
            imgs.append(img_01)
        ch = 3
        self.vis.images(imgs, nrow=ch, padding=10, opts=dict(title=labels), win = idx,env=self.name)
    def plot_current_errors(self, errors, interval,start):#error是个字典 {'error_type': error_value}

        for k,v in errors.items():
            if k not in self.plot_data.keys():
                self.plot_data[k] = []
                self.plot_data[k + '_x'] = []

            self.plot_data[k + '_x'].append(interval*len(self.plot_data[k])+start)
            self.plot_data[k].append(v)

            X = np.array(self.plot_data[k + '_x'])
            Y = np.array(self.plot_data[k])
            self.vis.line(
                X = X,
                Y = Y,
                opts={
                    'title': self.name + '   ' + k,
                    'legend': ['loss'],
                    'xlabel': 'iteration',
                    'ylabel': 'loss'},
                win = k,
            env=self.name)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, epochtime,i, errors, itertime, resttime):#t表示一次iter的时间
        rest_days = int(resttime/(3600*24))
        rest_hours = int((resttime-rest_days*24*3600)/3600)
        rest_min = int((resttime-rest_days*24*3600-rest_hours*3600)/60)
        rest_sec = int(resttime-rest_days*24*3600-rest_hours*3600-rest_min*60)
        self.message = '(epoch: %d, epochtime: %.3f,iters: %d, itertime: %.3f, resttime: %d days %d:%d:%d ) ' % (
        epoch, epochtime, i, itertime, rest_days, rest_hours, rest_min, rest_sec)

        for k, v in errors.items():
            self.message += '%s: %.6f ' % (k, v)
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
        self.message += '   date:' + nowTime
        print(self.message)

    def log_current_errors(self):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % self.message)

