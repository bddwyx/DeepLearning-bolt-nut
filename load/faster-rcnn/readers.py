from PIL import Image
import numpy as np
import data_Utilss
import json
from config import cfg
import cv2
import os

def custom_reader(data_dir, mode):

    def reader():
        file_list = open(data_dir)
        label_dict = {'bolt': 1, 'nut': 2}
        for line in file_list:#一个line就是train.txt的一行数据
            parts = line.split('\t')
            img_path = parts[0]
            batch_out = []
            if mode == 'train' or mode == 'eval':
                ######################  以下可能是需要自定义修改的部分   ############################
                img_id = [parts[0].split('/')[-1][:-4]]
                is_crowd = [0]
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size

                gt_cls = []
                gt_box = []
                crowd = []  #目标是否密集，一般为0
                for object_str in parts[1:]:
                    if len(object_str) <= 1:
                        continue

                    object = json.loads(object_str)
                    gt_cls.append(float(label_dict[object['value']])) #类别
                    bbox = object['coordinate']#坐标x1,y1,x2,y2
                    box = [float(bbox[0][0]), float(bbox[0][1]), float(bbox[1][0]), float(bbox[1][1])]
                    gt_box.append(box)
                    crowd.append(0)
                ######################  可能需要自定义修改部分结束   ############################
                img, im_scales = data_Utilss.get_image_blob(img_path, mode) #对图片做预处理
                c, h, w=img.shape
                img_info=[h, w, im_scales] #im_scales为图片伸缩尺寸
                outs=(np.array(img), np.array(gt_box, dtype = 'float32'), np.array(gt_cls, dtype = 'int32'), np.array(crowd, dtype = 'int32'), np.array(img_info, dtype = 'float32'), np.array(img_id, dtype = 'int64'))
                batch_out.append(outs)
                yield batch_out
            
    return reader
    
    
def infer(file_path):  #由图片路径返回图片和图片info
    def reader():
        if not os.path.exists(file_path):
            raise ValueError("Image path [%s] does not exist." % (file_path))
        im = cv2.imread(file_path)
        im = im.astype(np.float32, copy=False)
        im -= cfg.pixel_means
        im_height, im_width, channel = im.shape
        channel_swap = (2, 0, 1)  #(channel, height, width)
        im = im.transpose(channel_swap)
        im_info = np.array([im_height, im_width, 1.0], dtype=np.float32)
        yield [(im, im_info)]

    return reader