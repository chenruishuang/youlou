#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 19:51:59 2018

@author: yan
"""

import os
import cv2
import numpy as np
import pickle
import argparse
import torch
import gc
import time

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils


from utils.timer import Timer
from datasets.pascal_voc import VOCDataset
import cfgs.config as cfg

from test_me_map_newf import test_net
import ipdb


imdb_name = cfg.imdb_test
# trained_model = cfg.trained_model
#trained_model = os.path.join(cfg.train_output_dir,
#                             'darknet19_voc07trainval_exp3_75.h5')
#output_dir = cfg.test_output_dir

#yuanlaishi 300
max_per_image = 100
thresh = 0.01
vis = False

origin_time = 0

while True :
    
#    ipdb.set_trace()
    file_dir = './models/training/darknet19_voc07trainval_exp3/'
    file_dict = {}
    lists = os.listdir(file_dir) #先获取文件夹内的所有文件
    for i in lists: # 遍历所有文件
        ctime = os.stat(os.path.join(file_dir, i)).st_ctime
        file_dict[ctime] = i # 添加创建时间和文件名到字典
    max_ctime = max(file_dict.keys()) # 取值最大的时间
    
    if max_ctime > origin_time :
        origin_time = max_ctime
        print('start_write_and_test')
        print('newfile_time',max_ctime)
        print('file',file_dict[max_ctime])
        
        test_f = open('test_ap_newmodel.txt',mode = 'a+')
        test_f.writelines(['MODELS','\n',file_dict[max_ctime],'\n'])
        test_f.close()
        
        new_mdoels_path = os.path.join(file_dir,file_dict[max_ctime])
        imdb_map_newf = VOCDataset(imdb_name, cfg.DATA_DIR, cfg.batch_size,
                      yolo_utils.preprocess_test,
                      processes=1, shuffle=False, dst_size=cfg.multi_scale_inp_size)
        net = Darknet19()
        net_utils.load_net(new_mdoels_path, net)
        net.cuda()
        net.eval()
        
        test_net(net, imdb_map_newf, max_per_image, thresh, vis=False)
        print('test_this_models_done.....')
        
        imdb_map_newf.close()
        torch.cuda.empty_cache()
        gc.collect()
    time.sleep(1200)
#    print(file_dict[max_ctime]) #打印出最新文件名
#    print(max_ctime)