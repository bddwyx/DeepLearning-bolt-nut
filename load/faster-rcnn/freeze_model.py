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



def freeze():
    args = parse_args()
    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
    class_nums = cfg.class_num

    model = model_builder.RCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=False,
        mode='infer')
    model.build_model(image_shape)
    image,im_info=model.return_input()
    pred_boxes = model.eval_bbox_out()
    
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
#########3
    if not os.path.exists(cfg.pretrained_model):
        raise ValueError("Model path [%s] does not exist." % (cfg.pretrained_model))

    def if_exist(var):
        return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)

    fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    fluid.io.save_inference_model(dirname=args.freeze_model_save_dir, feeded_var_names=['image','im_info'], target_vars=pred_boxes, executor=exe)
    
    print('freeze succeed')

    # yapf: enable
    


if __name__ == '__main__':
    freeze()
