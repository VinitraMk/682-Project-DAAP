from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils

class ImageNetDataset(Dataset):
    length = 0
    
    def __init__(self, labels_path, data_dir, transform = None, lbl_mapping = {}):
        self.image_labels = pd.read_csv(labels_path)
        self.data_dir = data_dir
        self.transform = transform
        self.lbl_mapping = lbl_mapping

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.data_dir, self.image_labels.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.image_labels.iloc[idx, 1]
        sample = { 'image': image, 'label': label, 'oglbl': self.lbl_mapping[label], 'filename': self.image_labels.iloc[idx, 0] }
        if self.transform:
            sample = self.transform(sample)
        return sample
