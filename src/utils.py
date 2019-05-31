# ---------------------------------------------------------
# Tensorflow U-Net Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import cv2
import logging
import elasticdeform
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.ndimage import rotate


def init_logger(log_dir, name, is_train):
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)

    file_handler, stream_handler = None, None
    if is_train:
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

        # file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, name + '.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger, file_handler, stream_handler


def release_handles(logger, file_handler, stream_handler):
    file_handler.close()
    stream_handler.close()
    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)


def make_folders(is_train, cur_time=None):
    if is_train:
        model_dir = os.path.join('model', '{}'.format(cur_time))
        log_dir = os.path.join('logs', '{}'.format(cur_time))
        sample_dir = os.path.join('sample', '{}'.format(cur_time))

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)
    else:
        model_dir = os.path.join('model', '{}'.format(cur_time))
        log_dir = os.path.join('logs', '{}'.format(cur_time))
        sample_dir = os.path.join('sample', '{}'.format(cur_time))

    return model_dir, log_dir, sample_dir


def imshow(img, label, wmap, idx, alpha=0.6, delay=1, log_dir=None, show=False):
    img_dir = os.path.join(log_dir, 'img')
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    if len(img.shape) == 2:
        img = np.dstack((img, img, img))

    pseudo_label = None
    if len(label.shape) == 2:
        pseudo_label = pseudoColor(label)

    beta = 1. - alpha
    overlap = cv2.addWeighted(src1=img,
                              alpha=alpha,
                              src2=pseudo_label,
                              beta=beta,
                              gamma=0.0)

    # Convert to pseudo color map from gray-scale image
    wmap_color = cv2.applyColorMap(normalize_uint8(wmap), cv2.COLORMAP_JET)

    canvas = np.hstack((img, pseudo_label, overlap, wmap_color))
    cv2.imwrite(os.path.join(img_dir, 'GT_' + str(idx).zfill(2) + '.png'), canvas)

    if show:
        cv2.imshow('Show', canvas)

        if cv2.waitKey(delay) & 0xFF == 27:
            sys.exit('Esc clicked!')

def normalize_uint8(x, x_min=0, x_max=12, fit=255):
    x_norm = np.uint8(fit * (x - x_min) / (x_max - x_min))
    return x_norm


def pseudoColor(label, thickness=3):
    img = label.copy()
    img, contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = np.dstack((img, img, img))

    for i in range(len(contours)):
        cnt = contours[i]
        cv2.drawContours(img, [cnt], contourIdx=-1, color=(0, 255, 0), thickness=thickness)
        cv2.fillPoly(img, [cnt], color=randomColors(i))

    return img


def randomColors(idx):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]
    DarkRed = [0, 0, 139]
    PaleVioletRed = [147, 112, 219]
    Orange = [0, 165, 255]
    Teal = [128, 128, 0]

    color_dict = [Sky, Building, Pole, Road, Pavement,
                  Tree, SignSymbol, Fence, Car, Pedestrian,
                  Bicyclist, DarkRed, PaleVioletRed, Orange, Teal]

    return color_dict[idx % len(color_dict)]


def test_augmentation(img, label, wmap, idx, margin=10, log_dir=None):
    img_dir = os.path.join(log_dir, 'img')
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    img_tran, label_tran, wmap_tran = aug_translate(img, label, wmap)       # random translation
    img_flip, label_flip, wmap_flip = aug_flip(img, label, wmap)            # random horizontal and vertical flip
    img_rota, label_rota, wmap_rota = aug_rotate(img, label, wmap)          # random rotation
    img_defo, label_defo, wmap_defo = aug_elastic_deform(img, label, wmap)  # random elastic deformation
    # img_pert, label_pert, wmap_pert = aug_perturbation(img, label, wmap)  # random intensity perturbation

    # Arrange the images in a canvas and save them into the log file
    imgs = [img, img_tran, img_flip, img_rota, img_defo]
    labels = [label, label_tran, label_flip, label_rota, label_defo]
    wmaps = [wmap, wmap_tran, wmap_flip, wmap_rota, wmap_defo]
    h, w = img.shape
    canvas = np.zeros((3 * h + 4 * margin, len(imgs) * w + (len(imgs) + 1) * margin), dtype=np.uint8)

    for i, (img, label, wmap) in enumerate(zip(imgs, labels, wmaps)):
        canvas[1*margin:1*margin+h, (i+1) * margin + i * w:(i+1) * margin + (i + 1) * w] = img
        canvas[2*margin+1*h:2*margin+2*h, (i+1) * margin + i * w:(i+1) * margin + (i + 1) * w] = label
        canvas[3*margin+2*h:3*margin+3*h, (i+1) * margin + i * w:(i+1) * margin + (i + 1) * w] = normalize_uint8(wmap)

    cv2.imwrite(os.path.join(img_dir, 'augmentation_' + str(idx).zfill(2) + '.png'), canvas)


def test_cropping(img, label, idx, input_size, output_size, log_dir=None,
                  white=(255, 255, 255), blue=(255, 141, 47), red=(91, 70, 246), thickness=2, margin=10):
    img_dir = os.path.join(log_dir, 'img')
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    img_crop, label_crop, img_pad, rand_pos_h, rand_pos_w = cropping(
        img, label, input_size, output_size, is_extend=True)
    border_size = int((input_size - output_size) * 0.5)

    # Convert gray images to BGR images
    img_pad = np.dstack((img_pad, img_pad, img_pad))
    label_show = np.dstack((label, label, label))

    # Draw boundary lines
    img_pad = cv2.line(img=img_pad,
                       pt1=(0, border_size),
                       pt2=(img_pad.shape[1]-1, border_size),
                       color=white,
                       thickness=thickness)
    img_pad = cv2.line(img=img_pad,
                       pt1=(0, img_pad.shape[0]-1-border_size),
                       pt2=(img_pad.shape[1]-1, img_pad.shape[0]-1-border_size),
                       color=white,
                       thickness=thickness)
    img_pad = cv2.line(img=img_pad,
                       pt1=(border_size, 0),
                       pt2=(border_size, img_pad.shape[0]-1),
                       color=white,
                       thickness=thickness)
    img_pad = cv2.line(img=img_pad,
                       pt1=(img_pad.shape[1]-1-border_size, 0),
                       pt2=(img_pad.shape[1]-1-border_size, img_pad.shape[0]-1),
                       color=white,
                       thickness=thickness)

    # Draw the ROI input region
    img_pad = cv2.rectangle(img=img_pad,
                            pt1=(rand_pos_w+border_size, rand_pos_h+border_size),
                            pt2=(rand_pos_w+border_size+output_size, rand_pos_h+border_size+output_size),
                            color=red,
                            thickness=thickness+1)
    img_pad = cv2.rectangle(img=img_pad,
                            pt1=(rand_pos_w, rand_pos_h),
                            pt2=(rand_pos_w+input_size, rand_pos_h+input_size),
                            color=blue,
                            thickness=thickness+1)
    label_show = cv2.rectangle(img=label_show,
                               pt1=(rand_pos_w, rand_pos_h),
                               pt2=(rand_pos_w+output_size, rand_pos_h+output_size),
                               color=red,
                               thickness=thickness+1)

    img_crop = cv2.rectangle(img=np.dstack((img_crop, img_crop, img_crop)),
                             pt1=(2, 2),
                             pt2=(img_crop.shape[1]-2, img_crop.shape[0]-2),
                             color=blue,
                             thickness=thickness+1)
    label_crop = cv2.rectangle(img=np.dstack((label_crop, label_crop, label_crop)),
                               pt1=(2, 2),
                               pt2=(label_crop.shape[1]-2, label_crop.shape[0]-2),
                               color=red,
                               thickness=thickness+1)

    canvas = np.zeros((img_pad.shape[0] + label_show.shape[0] + 3 * margin,
                       img_pad.shape[1] + img_crop.shape[1] + 3 * margin, 3), dtype=np.uint8)

    # Copy img_pad
    h_start = margin
    w_start = margin
    canvas[h_start:h_start + img_pad.shape[0], w_start:w_start+img_pad.shape[1], :] = img_pad

    # Copy label_show
    h_start = 2 * margin + img_pad.shape[0]
    w_start = margin
    canvas[h_start:h_start + label_show.shape[0], w_start:w_start + label_show.shape[0], :] = label_show

    # Draw connections between the left and right images
    # Four connections for the upper images
    canvas = cv2.line(img=canvas,
                      pt1=(margin+rand_pos_w, margin+rand_pos_h),
                      pt2=(2*margin+img_pad.shape[1], margin),
                      color=blue,
                      thickness=thickness+1)
    canvas = cv2.line(img=canvas,
                      pt1=(margin+rand_pos_w+input_size, margin+rand_pos_h),
                      pt2=(2*margin+img_pad.shape[1]+img_crop.shape[1], margin),
                      color=blue,
                      thickness=thickness+1)
    canvas = cv2.line(img=canvas,
                      pt1=(margin+rand_pos_w, margin+rand_pos_h+input_size),
                      pt2=(2*margin+img_pad.shape[1], margin+input_size),
                      color=blue,
                      thickness=thickness+1)
    canvas = cv2.line(img=canvas,
                      pt1=(margin+rand_pos_w+input_size, margin+rand_pos_h+input_size),
                      pt2=(2*margin+img_pad.shape[1]+img_crop.shape[1], margin+input_size),
                      color=blue,
                      thickness=thickness+1)

    # Four connections for the bottom images
    canvas = cv2.line(img=canvas,
                      pt1=(margin+rand_pos_w, 2*margin+img_pad.shape[1]+rand_pos_h),
                      pt2=(2*margin+img_pad.shape[0], 2*margin+img_pad.shape[1]),
                      color=red,
                      thickness=thickness+1)
    canvas = cv2.line(img=canvas,
                      pt1=(margin+output_size+rand_pos_w, 2*margin+img_pad.shape[0]+rand_pos_h),
                      pt2=(2*margin+img_pad.shape[1]+output_size, 2*margin+img_pad.shape[0]),
                      color=red,
                      thickness=thickness+1)
    canvas = cv2.line(img=canvas,
                      pt1=(margin+rand_pos_w, 2*margin+img_pad.shape[0]+output_size+rand_pos_h),
                      pt2=(2*margin+img_pad.shape[1], 2*margin+img_pad.shape[0]+output_size),
                      color=red,
                      thickness=thickness+1)
    canvas = cv2.line(img=canvas,
                      pt1=(margin+rand_pos_w+output_size, 2*margin+img_pad.shape[1]+output_size+rand_pos_h),
                      pt2=(2*margin+img_pad.shape[1]+output_size, 2*margin+img_pad.shape[1]+output_size),
                      color=red,
                      thickness=thickness+1)

    # Copy img_crop
    h_start = margin
    w_start = 2 * margin + img_pad.shape[1]
    canvas[h_start:h_start + img_crop.shape[0], w_start:w_start + img_crop.shape[1], :] = img_crop

    # Copy label_crop
    h_start = 2 * margin + img_pad.shape[0]
    w_start = 2 * margin + img_pad.shape[1]
    canvas[h_start:h_start + label_crop.shape[0], w_start:w_start + label_crop.shape[1], :] = label_crop

    cv2.imwrite(os.path.join(img_dir, 'crop_' + str(idx).zfill(2) + '.png'), canvas)


def aug_translate(img, label, wmap, max_factor=1.2):
    assert len(img.shape) == 2 and len(label.shape) == 2 and len(wmap.shape) == 2

    # Resize originl image
    resize_factor = np.random.uniform(low=1.,  high=max_factor)
    img_bigger = cv2.resize(src=img.copy(), dsize=None, fx=resize_factor, fy=resize_factor,
                            interpolation=cv2.INTER_LINEAR)
    label_bigger = cv2.resize(src=label.copy(), dsize=None, fx=resize_factor, fy=resize_factor,
                              interpolation=cv2.INTER_NEAREST)
    wmap_bigger = cv2.resize(src=wmap.copy(), dsize=None, fx=resize_factor, fy=resize_factor,
                             interpolation=cv2.INTER_NEAREST)

    # Generate random positions for horizontal and vertical axes
    h_bigger, w_bigger = img_bigger.shape
    h_star = np.random.random_integers(low=0, high=h_bigger-img.shape[0])
    w_star = np.random.random_integers(low=0, high=w_bigger-img.shape[1])

    # Crop image from the bigger one
    img_crop = img_bigger[h_star:h_star+img.shape[1], w_star:w_star+img.shape[0]]
    label_crop = label_bigger[h_star:h_star+img.shape[1], w_star:w_star+img.shape[0]]
    wmap_crop = wmap_bigger[h_star:h_star+img.shape[1], w_star:w_star+img.shape[0]]

    return img_crop, label_crop, wmap_crop


def aug_flip(img, label, wmap):
    assert len(img.shape) == 2 and len(label.shape) == 2 and len(wmap.shape) == 2

    # Random horizontal flip
    if np.random.uniform(low=0., high=1.) > 0.5:
        img_hflip = cv2.flip(src=img, flipCode=0)
        label_hflip =  cv2.flip(src=label, flipCode=0)
        wmap_hflip = cv2.flip(src=wmap, flipCode=0)
    else:
        img_hflip = img.copy()
        label_hflip = label.copy()
        wmap_hflip = wmap.copy()

    # Random vertical flip
    if np.random.uniform(low=0., high=1.) > 0.5:
        img_vflip = cv2.flip(src=img_hflip, flipCode=1)
        label_vflip = cv2.flip(src=label_hflip, flipCode=1)
        wmap_vflip = cv2.flip(src=wmap_hflip, flipCode=1)
    else:
        img_vflip = img_hflip.copy()
        label_vflip = label_hflip.copy()
        wmap_vflip = wmap_hflip.copy()

    return img_vflip, label_vflip, wmap_vflip


def aug_rotate(img, label, wmap):
    assert len(img.shape) == 2 and len(label.shape) == 2 and len(wmap.shape)

    # Random rotate image
    angle = np.random.randint(low=0, high=360, size=None)
    img_rotate = rotate(input=img, angle=angle, axes=(0, 1), reshape=False, order=3, mode='reflect')
    label_rotate = rotate(input=label, angle=angle, axes=(0, 1), reshape=False, order=3, mode='reflect')
    wmap_rotate = rotate(input=wmap, angle=angle, axes=(0, 1), reshape=False, order=3, mode='reflect')

    # Correct label map
    ret, label_rotate = cv2.threshold(src=label_rotate, thresh=127.5, maxval=255, type=cv2.THRESH_BINARY)

    return img_rotate, label_rotate, wmap_rotate


def aug_elastic_deform(img, label, wmap):
    assert len(img.shape) == 2 and len(label.shape) == 2 and len(wmap.shape) == 2

    # Apply deformation with a random 3 x 3 grid to inputs X and Y,
    # with a different interpolation for each input
    img_distort, label_distort, wmap_distort = elasticdeform.deform_random_grid(X=[img, label, wmap],
                                                                                sigma=10,
                                                                                points=3,
                                                                                order=[3, 0, 0],
                                                                                mode='mirror')

    return img_distort, label_distort, wmap_distort


def aug_perturbation(img, label, wmap, low=0.8, high=1.2):
    pertur_map = np.random.uniform(low=low, high=high, size=img.shape)
    img_en = np.round(img * pertur_map).astype(np.uint8)
    img_en = np.clip(img_en, a_min=0, a_max=255)
    return img_en, label, wmap


def cropping(img, label, input_size, output_size, is_extend=False):
    border_size = int((input_size - output_size) * 0.5)
    rand_pos_h = np.random.randint(low=0, high=img.shape[0] - output_size)
    rand_pos_w = np.random.randint(low=0, high=img.shape[1] - output_size)

    img_pad = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_REFLECT_101)
    img_crop = img_pad[rand_pos_h:rand_pos_h+input_size, rand_pos_w:rand_pos_w+input_size].copy()
    label_crop = label[rand_pos_h:rand_pos_h+output_size, rand_pos_w:rand_pos_w+output_size].copy()

    if is_extend:
        return img_crop, label_crop, img_pad, rand_pos_h, rand_pos_w
    else:
        return img_crop, label_crop


def pre_bilaterFilter(img, d=3, sigmaColor=75, simgaSpace=75):
    pre_img = cv2.bilateralFilter(src=img, d=d, sigmaColor=sigmaColor, sigmaSpace=simgaSpace)
    return pre_img

def acc_measure(true_arr, pred_arr):
    cm = confusion_matrix(true_arr, pred_arr)
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    return acc
