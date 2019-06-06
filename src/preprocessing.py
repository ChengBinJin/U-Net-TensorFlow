# ---------------------------------------------------------
# Tensorflow U-Net Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

def main(dataset_path=None, is_write=False):
    # train_imgs = tiff.imread(os.path.join(dataset_path, 'train-volume.tif'))
    train_labels = tiff.imread(os.path.join(dataset_path, 'train-labels.tif'))

    save_dir = 'wmap_imgs'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    file_name = os.path.join(dataset_path, 'train-wmaps.npy')
    if is_write:
        wmaps = np.zeros_like(train_labels, dtype=np.float32)
        for idx in range(train_labels.shape[0]):
            print('Image index: {}'.format(idx))
            img = train_labels[idx]
            cal_weight_map(label=img, wmaps=wmaps, save_dir=save_dir, iter_time=idx)

        np.save(file_name, wmaps)

    wmaps = np.load(file_name)
    plot_wmaps(wmaps)


def plot_wmaps(wmaps, nrows=5, ncols=6, hspace=0.2, wspace=0.1, vmin=0., vmax=12., interpolation='nearest'):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))

    # Adjust vertical and horizontal spacing
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    im = None
    for i, ax in enumerate(axes.flat):
        if i < (nrows * ncols):
            # Plot image
            im = ax.imshow(wmaps[i], interpolation=interpolation, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)

            # Show the classes as the label on the x-axis.
            xlabel = "Weight Map: {0}".format(str(i).zfill(2))
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

            # Add colorbar
            # fig.colorbar(im)  # cmap=plt.cm.jet, vmin=vmin, vmax=vmax,

    # fig.subplots_adjust(right=0.8)
    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()


def cal_weight_map(label, wmaps, save_dir, iter_time, wc=1., w0=10., sigma=5, interval=500, vmin=0, vmax=12):
    _, contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wmap = wc * np.ones_like(label, dtype=np.float32)
    img = label.copy()

    y_points, x_points = np.where(img == 0)
    for idx, (y_point, x_point) in enumerate(zip(y_points, x_points)):
        if np.mod(idx, interval) == 0:
            print('{} / {}'.format(idx, len(y_points)))

        point = np.array([x_point, y_point]).astype(np.float32)
        dis_arr = []

        for i in range(len(contours)):
            cnt = (np.squeeze(contours[i])).astype(np.float32)
            dis_arr.append(np.amin(np.sqrt(np.sum(np.power(point - cnt, 2), axis=1))))

        dis_arr.sort()  # sorting
        wmap[y_point, x_point] += wc + w0 * np.exp(- np.power(np.sum(dis_arr[0:2]), 2) / (2 * sigma * sigma))

    plt.imshow(wmap, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    plt.axis('off')

    # To solve the multiple color-bar problem
    if iter_time == 0:
        plt.colorbar()

    plt.savefig(os.path.join(save_dir, str(iter_time).zfill(2) + '.png'), bbox_inches='tight')
    wmaps[iter_time] = wmap


if __name__ == '__main__':
    main(dataset_path='../../Data/EMSegmentation', is_write=False)


