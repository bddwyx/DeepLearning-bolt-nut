from config import cfg
import numpy as np
import cv2


def get_image_blob(image_path, mode):
    """
    对图片做预处理，伸缩图片。
    """
    if mode == 'train':
        scales = cfg.TRAIN.scales
        scale_ind = np.random.randint(0, high=len(scales))
        target_size = scales[scale_ind]
        max_size = cfg.TRAIN.max_size
    else:
        target_size = cfg.TEST.scales[0]
        max_size = cfg.TEST.max_size
    im = cv2.imread(image_path)
    try:
        assert im is not None
    except AssertionError as e:
        print('Failed to read image \'{}\''.format(image_path))
        os._exit(0)
    im, im_scale = prep_im_for_blob(im, cfg.pixel_means, target_size, max_size)

    return im, im_scale


def prep_im_for_blob(im, pixel_means, target_size, max_size):#图像减均值，并resize到：最小边到target size时最大边不超过max size 或 最大边到max size时最小边不超过target size
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    im_height, im_width, channel = im.shape
    channel_swap = (2, 0, 1)  #(batch, channel, height, width)
    im = im.transpose(channel_swap)
    return im, im_scale


def padding_minibatch(batch_data):  # 将图像放大到最大尺寸，不足的地方补0
    if len(batch_data) == 1:
        return batch_data

    max_shape = np.array([data[0].shape for data in batch_data]).max(axis=0)

    padding_batch = []
    for data in batch_data:
        im_c, im_h, im_w = data[0].shape[:]
        padding_im = np.zeros(
            (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = data[0]
        padding_batch.append((padding_im,) + data[1:])
    return padding_batch