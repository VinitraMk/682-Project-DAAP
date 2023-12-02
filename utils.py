import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import pandas as pd
from sklearn import preprocessing
from skimage import io

# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
def dataloader(train_size, test_size, data_dir, batch_size, num_workers, total_num=50000):
    # Setup the transformation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    index = np.arange(total_num)
    np.random.shuffle(index)
    train_index = index[:train_size]
    test_index = index[train_size: (train_size + test_size)]

    train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index), num_workers=num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_index), num_workers=num_workers, pin_memory=True, shuffle=False)
    return train_loader, test_loader

def make_csv(data_dir):
    data_dir_abs = os.path.join(os.getcwd(), data_dir)
    cols = ['filename', 'label', 'filepath']
    train_df = pd.DataFrame([], columns = cols)
    val_df = pd.DataFrame([], columns = cols)

    le = preprocessing.LabelEncoder()
    le.fit(os.listdir(os.path.join(data_dir_abs, 'train')))
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    lbl_le_mapping = dict(zip(le.transform(le.classes_), le.classes_))


    for split in ['train', 'val']:
        sub_dir_path = os.path.join(data_dir_abs, split)
        labels = os.listdir(sub_dir_path)
        data = []
        for lbl in labels:
            label_dir = os.path.join(sub_dir_path, lbl)
            data += [(fname, le_name_mapping[lbl], f'{lbl}/{fname}') for fname in os.listdir(label_dir)]
        if split == 'train':
            train_df = pd.concat([pd.DataFrame(data, columns = train_df.columns), train_df], ignore_index = True)
        else:
            val_df = pd.concat([pd.DataFrame(data, columns = val_df.columns), val_df], ignore_index = True)
        train_df.to_csv(os.path.join(data_dir_abs, 'train_images.csv'), index = False)
        val_df.to_csv(os.path.join(data_dir_abs, 'val_images.csv'), index = False)
    return lbl_le_mapping

def imshow(img, transforms=None, figsize=(10, 20)):
    npimg = _tensor_to_show(img, transforms)
    plt.figure(figsize=figsize)
    plt.imshow(npimg, interpolation=None)