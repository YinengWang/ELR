import sys

import numpy as np
from PIL import Image
import torchvision
import torch
import pandas as pd
import os
from torch.utils.data import Dataset
import math

def get_cdon(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None):


    root_folder = "/home/dd2424-google/Supervised-Image-Classification-with-Noisy-Labels-Using-Deep-Learning/Datasets/CDON"
    
    base_dataset = CDONdataset("dataset_lables.csv", root_folder, transform=transform_train)
    
    train_dataset = CDONDatasetSplit(root, cfg_trainer, base_dataset, split=0.9, from_bottom=True)
    val_dataset = CDONDatasetSplit(root, cfg_trainer, base_dataset, split=0.1, from_bottom=False)   
    
    
    return train_dataset, val_dataset

        
class CDONdataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.sublabels_mapping = {} # the sublabels have an arbitrary number, use this to enumerate them
        self.root_dir = root_dir
        self.transform = transform
        csv_path = os.path.join(self.root_dir, csv_file)
        self.annotations = pd.read_csv(csv_path, delimiter=',', header=None)

    def __len__(self):
        return int(len(self.annotations))

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "images", self.annotations.iloc[index, 0])
        with Image.open(img_path).convert('RGB') as image:
            y_wild_label = self.annotations.iloc[index, 2]
            if y_wild_label == y_wild_label:
                y_label = int(y_wild_label)
            else:
                y_label = 0
            if not y_label in self.sublabels_mapping:
                self.sublabels_mapping[y_label] = len(self.sublabels_mapping) + 1
                y_label = len(self.sublabels_mapping)
            else:
                y_label = self.sublabels_mapping[y_label]
            # y_label = torch.tensor(int(self.annotations.iloc[index, 1])) - 1 # this is the main category
            if self.transform:
                image = self.transform(image)

        return (image, torch.tensor(y_label))

# Used to split the above dataset into two (possibly for training / testing)
class CDONDatasetSplit(Dataset):
    def __init__(self, root, cfg_trainer, dataset: Dataset, split, from_bottom=True, samples4category=1000):
        self.num_classes = 64
        self.cfg_trainer = cfg_trainer
        self.samples4category = samples4category
        self.original_dataset = dataset
        self.split = split
        self.from_bottom = from_bottom

    def __len__(self):
        if self.from_bottom:
            return math.floor(len(self.original_dataset) * self.split)
        return math.ceil(len(self.original_dataset) * self.split)

    def __getitem__(self, index):
        original_index = math.floor(index / self.samples4category / self.split) * self.samples4category
        samples_missing = len(self.original_dataset) - original_index
        
        if not self.from_bottom:
            if samples_missing < self.samples4category:
                # we are at the end of the dataset
                original_index += int(samples_missing * (1 - self.split))
            else:
                original_index += int(self.samples4category * (1 - self.split))
        original_index += (index % (self.samples4category * self.split))
        ret = self.original_dataset[int(original_index)]
        return ret[0], ret[1], index, 0
