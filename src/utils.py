# ---------------------------------------------------------
# Tensorflow U-Net Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import sys
import numpy as np


def make_folders(is_train, cur_time=None):
    if is_train:
        model_dir = os.path.join('model', '{}'.format(cur_time))
        log_dir = os.path.join('logs', '{}'.format(cur_time))

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    else:
        model_dir = os.path.join('model', '{}'.format(cur_time))
        log_dir = os.path.join('logs', '{}'.format(cur_time))

    return model_dir, log_dir


def imshow(img, label, idx, alpha=0.6, delay=1, log_dir=None):
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

    canvas = np.hstack((img, pseudo_label, overlap))
    cv2.imshow('Show', canvas)

    if cv2.waitKey(delay) & 0xFF == 27:
        sys.exit('Esc clicked!')

    cv2.imwrite(os.path.join(log_dir, str(idx).zfill(2) + '.png'), canvas)

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
