import torch
import torchvision
import tqdm
import os
import numpy as np
import gzip
import pickle
from torchvision.io import read_image, write_png

import utils
import config as C

def unnormalize(img):
    m = torch.Tensor([0.485, 0.456, 0.406])
    s = torch.Tensor([0.229, 0.224, 0.225])
    m = m.unsqueeze(1).unsqueeze(1)
    s = s.unsqueeze(1).unsqueeze(1)
    return img * s + m

def attack_folder(raw_img_folder, save_folder, args, zero_out=False):
    with gzip.open("data/imagenet_patch.gz", 'rb') as f:
        patches, targets, class_dict = pickle.load(f)
    dataset = utils.ImageDataset(
        raw_img_folder
    )
    data_classid2name = {
        v:k for k, v in dataset.class_to_idx.items()
    }
    for img, label, img_name in tqdm.tqdm(dataset, desc="Attacking"):
        patch_num = np.random.randint(10)
        patch = patches[patch_num]
        if zero_out:
            patch = 0 * patch
        apply_patch = utils.ApplyPatch(
            patch,
            translation_range=(0.3, 0.3),    # translation fraction wrt image dimensions
            rotation_range=10,             # maximum absolute value of the rotation in degree
            scale_range=(1, 1.2)           # scale range wrt image dimensions
        )
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            apply_patch,
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        label = C.CLASS_NAME2ID[data_classid2name[label]]

        img = transforms(img)
        img = 255*unnormalize(img)

        os.makedirs(os.path.join(save_folder, str(C.CLASS_ID2NAME[label])), exist_ok=True)
        write_png(img.to(torch.uint8), os.path.join(save_folder, str(C.CLASS_ID2NAME[label]), img_name))


