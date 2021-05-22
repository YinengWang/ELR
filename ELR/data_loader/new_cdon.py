import torchvision

import numpy as np
from PIL import Image
import torchvision
import torch
import pandas as pd
import os
from torch.utils.data import Dataset
import math
import pickle


def get_new_cdon(cfg_trainer,
             transform_train, transform_val):
    data, labels = load_cdon()
    labels = torch.tensor(labels, dtype=torch.long)

    train_idxs, val_idxs = train_val_split(labels)
    train_dataset = CDON_train(data, labels, cfg_trainer, train_idxs, transform=transform_train)
    val_dataset = CDON_train(data, labels, cfg_trainer, val_idxs, transform=transform_val)

    return train_dataset, val_dataset


def load_cdon(root_dir="/home/dd2424-google/Supervised-Image-Classification-with-Noisy-Labels-Using-Deep-Learning/Datasets/CDON/"):
    if os.path.exists(os.path.join(root_dir, 'data', 'cdon_data.pkl')):
        with open(os.path.join(root_dir, 'data', 'cdon_data.pkl'), 'rb') as f:
            return pickle.load(f)
    annotations = pd.read_csv(os.path.join(root_dir, 'dataset_lables.csv'), delimiter=',', header=None)
    data = []
    for filename in annotations.iloc[:,0]:
        img_path = os.path.join(root_dir, "images", filename)
        with Image.open(img_path).convert('RGB') as image:
            data.append(np.array(image))
    labels = pd.Categorical(annotations.iloc[:,2]).codes
    with open(os.path.join(root_dir, 'data', 'cdon_data.pkl'), 'wb') as f:
        pickle.dump((data, labels), f)
    return data, labels


def train_val_split(base_dataset: torchvision.datasets.CIFAR10):
    num_classes = 63
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CDON_train(torchvision.datasets.CIFAR10):
    def __init__(self, data, labels, cfg_trainer, indexs,
                 transform=None, target_transform=None,
                 download=True):
        super(CDON_train, self).__init__('./data', train=False,
                                         transform=transform, target_transform=target_transform,
                                         download=download)
        self.num_classes = 63
        self.data, self.targets = data, labels
        self.cfg_trainer = cfg_trainer
        self.train_data = [self.data[i] for i in indexs]  # self.train_data[indexs]
        self.train_labels = np.array(self.targets)[indexs]  # np.array(self.train_labels)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        self.train_labels_gt = self.train_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt

    def __len__(self):
        return len(self.train_data)


# class CDON_val(torchvision.datasets.CIFAR10):
#
#     def __init__(self, data, labels, cfg_trainer, indexs, train=True,
#                  transform=None, target_transform=None,
#                  download=False):
#         super(CDON_val, self).__init__('./data', train=train,
#                                           transform=transform, target_transform=target_transform,
#                                           download=download)
#
#         self.data, self.targets = data, labels
#         # self.train_data = self.data[indexs]
#         # self.train_labels = np.array(self.targets)[indexs]
#         self.num_classes = 63
#         self.cfg_trainer = cfg_trainer
#         if train:
#             self.train_data = self.data[indexs]
#             self.train_labels = np.array(self.targets)[indexs]
#         else:
#             self.train_data = self.data
#             self.train_labels = np.array(self.targets)
#         self.train_labels_gt = self.train_labels.copy()
#
#     def __len__(self):
#         return len(self.train_data)
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, index, target_gt
