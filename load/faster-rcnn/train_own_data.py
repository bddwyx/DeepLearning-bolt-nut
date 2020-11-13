#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


import sys
import numpy as np
import time
import shutil
from utility import parse_args, print_arguments, SmoothedValue, TrainingStats, now_time, check_gpu
import collections
import readers
import paddle
import paddle.fluid as fluid

import models.model_builder as model_builder
import models.resnet as resnet
from learning_rate import exponential_with_warmup_decay
from config import cfg


def train():
    learning_rate = cfg.learning_rate
    image_shape = [3, cfg.TRAIN.max_size, cfg.TRAIN.max_size]

    use_random = True
    
#################################################################  
    model = model_builder.RCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=cfg.use_pyreader,
        use_random=use_random)
    model.build_model(image_shape)
    losses, keys = model.loss()
    loss = losses[0]
    fetch_list = losses   #建立模型并传出loss
#################################################################
    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    lr = exponential_with_warmup_decay(
        learning_rate=learning_rate,
        boundaries=boundaries,
        values=values,
        warmup_iter=cfg.warm_up_iter,
        warmup_factor=cfg.warm_up_factor)#学习率设置
##################################################################
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)
    optimizer.minimize(loss)  #优化器设置
##################################################################
    fetch_list = fetch_list + [lr]

    for var in fetch_list:
        var.persistable = True

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
##################################################################
    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))

        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)  #加载预训练模型
##################################################################
    train_exe = exe
    
    data_dir =cfg.data_dir #data/data6045/train.txt
    train_reader = readers.custom_reader(data_dir, mode='train')
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())    #读取数据reader设置
##################################################################
    def save_model(postfix):###保存模型
        model_path = os.path.join(cfg.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(exe, model_path)
        
    def freeze_model(postfix):#####固化模型
        model_path = os.path.join(cfg.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_inference_model(dirname=model_path,
                                              feeded_var_names=image.name,
                                              target_vars=[pred_boxes],
                                              main_program=main_program.clone(for_test=True),
                                              executor=exe)
        
######################################################
    def train_loop():   #训练主体
        start_time = time.time()
        prev_start_time = start_time
        start = start_time
        train_stats = TrainingStats(cfg.log_window, keys)
        #epoch=500
        for i in range(cfg.epoch):
            for iter_id, data in enumerate(train_reader()):
                prev_start_time = start_time
                start_time = time.time()
                outs = train_exe.run(fetch_list=[v.name for v in fetch_list],
                                     feed=feeder.feed(data),return_numpy=False)
    
                stats = {k: np.array(v).mean() for k, v in zip(keys, outs[:-1])}
                train_stats.update(stats)
                logs = train_stats.log()
                # iter_id=iter_id + i*415   #415为每个epoch内的最大iter数，数据集图片数量/batch_size
                strs = '{},epoch: {}, iter: {}, lr: {:.5f}, {}, time: {:.3f}'.format(
                    now_time(), i,  iter_id,
                    np.mean(outs[-1]), logs, start_time - prev_start_time)
                if iter_id % 100 == 0:
                    print(strs)    #每隔100个iter打印一次结果
                sys.stdout.flush()

                if (iter_id + 1) == cfg.max_iter:   #达到最大设定的iter数就停止训练
                    print('break')
                    break
            save_model('model_final')   #每个epoch保存模型
            #freeze_model('model_freeze')
        end_time = time.time()
        total_time = end_time - start_time
        last_loss = np.array(outs[0]).mean()
###########################################################################     
    
    train_loop()
    save_model('model_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)  #打印超参数
    check_gpu(args.use_gpu)
    train()
