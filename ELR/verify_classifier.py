import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_loader.new_cdon import load_cdon


def show_img(ax, img):
    img = np.rot90(img, 1)
    im = img.reshape(img.shape, order='F')
    sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
    sim = sim.transpose(1, 0, 2)
    ax.imshow(sim, interpolation='nearest')
    ax.axis('off')


def verify_classifier(num_classes=63, num_imgs_per_cls=1):
    data, labels = load_cdon('../CDON-dataset/CDON')
    fig, ax = plt.subplots(num_imgs_per_cls, num_classes)
    labels = pd.Series(labels)
    for category in range(num_classes):
        indices = labels[labels == category].index[:num_imgs_per_cls]
        imgs = [data[idx] for idx in indices]
        if num_imgs_per_cls == 1:
            show_img(ax[category], imgs[0])
        else:
            for y, img in enumerate(imgs):
                show_img(ax[category][y], img)
    plt.show()


def show_imgs_by_categories(num_classes=63, num_imgs_per_cls=1):
    data, labels = load_cdon('../CDON-dataset/CDON')
    fig, ax = plt.subplots(8, 8)
    labels = pd.Series(labels)
    for category in range(num_classes):
        indices = labels[labels == category].index[:num_imgs_per_cls]
        imgs = [data[idx] for idx in indices]
        x, y = category // 8, category % 8
        show_img(ax[x][y], imgs[0])
    plt.show()


if __name__ == '__main__':
    # verify_classifier()
    show_imgs_by_categories()
