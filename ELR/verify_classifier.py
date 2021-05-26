from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data._utils import collate

from our_model.model import ResNet34
# from model.model import resnet34

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


def plot_category(ax, pred):
    ax.text(0, 0, str(pred))
    # ax.axis('off')


def verify_classifier(num_classes=63, num_imgs_per_cls=1):
    data, labels = load_cdon('/home/dd2424-msi/20210525-retest-ce/Datasets/CDON')
    model = ResNet34(num_classes=num_classes)
    model.load_state_dict(torch.load('/tmp/pycharm_project_295/CDON-dataset/cdon-model'))
    model.eval()
    transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0036, 0.0038, 0.0040), (0.0043, 0.0046, 0.0044)),
        ])
    #     .Compose([
    #     transforms.Resize(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.0036, 0.0038, 0.0040), (0.0043, 0.0046, 0.0044)),
    # ])
    fig, ax = plt.subplots(8, 8)
    labels = pd.Series(labels)
    for category in range(num_classes):
        indices = labels[labels == category].index[:num_imgs_per_cls]
        imgs = [data[idx] for idx in indices]
        img = Image.fromarray(imgs[0])
        inputs = collate.default_collate([transform(img)])
        with torch.no_grad():
            output = model(inputs)
            pred = torch.argmax(output, dim=1)
            # print(pred)
            top_3 = sorted(output, reverse=True)[:3]
        x, y = category // 8, category % 8
        show_img(ax[x][y], imgs[0])
        plot_category(ax[x][y], ', '.join(str(s) for s in top_3))
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
    verify_classifier()
    # show_imgs_by_categories()
