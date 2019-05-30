import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

def main(dataset_path=None):
    # train_imgs = tiff.imread(os.path.join(dataset_path, 'train-volume.tif'))
    train_labels = tiff.imread(os.path.join(dataset_path, 'train-labels.tif'))

    save_dir = 'wmap_imgs'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for idx in range(train_labels.shape[0]):
        print('Image index: {}'.format(idx))
        img = train_labels[idx]
        cal_weight_map(img, dataset_path, save_dir=save_dir, iter_time=idx)


def cal_weight_map(label, data_dir, save_dir, iter_time, wc=1., w0=10., sigma=5, interval=500, vmin=0, vmax=12):
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

    if iter_time == 0:
        plt.colorbar()

    plt.savefig(os.path.join(save_dir, str(iter_time).zfill(2) + '.png'), bbox_inches='tight')
    np.save(os.path.join(data_dir, 'train-wmap' + str(iter_time).zfill(2) + '.npy'), wmap)

if __name__ == '__main__':
    main(dataset_path='../../Data/EMSegmentation')


