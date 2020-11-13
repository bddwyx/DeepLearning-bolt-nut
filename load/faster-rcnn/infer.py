import os
import time
import numpy as np
from eval_helper import *
import paddle
import paddle.fluid as fluid

from utility import print_arguments, parse_args, check_gpu
import models.model_builder as model_builder
import models.resnet as resnet
from config import cfg
import readers



def freeze_infer():

    labels_map={0: 'background', 1: 'bolt', 2: 'nut'}
    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
    class_nums = cfg.class_num

    model = model_builder.RCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=False,
        mode='infer')
    model.build_model(image_shape)
    
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname='faster-rcnn/freeze_model', executor=exe)
    # yapf: enable
    
    infer_reader = readers.infer(cfg.image_path)
    feed_list=model.feeds()
    
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())
        
    data = next(infer_reader())
    
    result = exe.run(inference_program,
          feed=feeder.feed(data),
          fetch_list=fetch_targets,return_numpy=False)
    
    pred_boxes_v = result[0]
    
    new_lod = pred_boxes_v.lod()
    nmsed_out = pred_boxes_v
    image = None
    
    draw_bounding_box_on_image(cfg.image_path, nmsed_out, cfg.draw_threshold,
                               labels_map, image)


if __name__ == '__main__':
    args = parse_args()
    freeze_infer()
