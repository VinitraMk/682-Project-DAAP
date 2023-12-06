import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from transforms.random_affine import RandomAffine
from torchvision.utils import save_image


class PatchAttack(nn.Module):
    INPUT_SIZE = 224
    PATCH_SIZE = 50
    device = "cpu"
    use_cuda = False
    mode = 'train'
    output_dir = 'attacked-images'
    lbl_le_mapping = {}

    def __init__(self, model, output_dir, patch, translation_range=(0.2, 0.2), rotation_range=45,
                scale_range=(0.5, 1), patch_size=50, use_cuda = False):
        super(PatchAttack, self).__init__()
        self.model = model 
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.__transforms = None
        self.__patch = None
        self.__input_shape = None
        self.__mask = None
        self.set_transforms(translation_range, rotation_range, scale_range)
        self.set_patch(patch)
        if use_cuda:
            self.use_cuda = use_cuda
            self.device = "cuda"
            self.model.to(self.device)

    @property
    def mask(self):
       return self.__mask

    @property 
    def transforms(self):
       return self.__transforms
    
    def set_patch(self, patch):
       self.__patch = patch
       self.__input_shape = self.__patch.shape
       self.__mask = self.__generate_mask()
    
    def __generate_mask(self):
       mask = torch.ones(self.__input_shape)
       uppleft_x = self.__input_shape[2] // 2 - self.patch_size // 2
       uppleft_y = self.__input_shape[1] // 2 - self.patch_size // 2
       bottright_x = self.__input_shape[2] // 2 + self.patch_size // 2
       bottright_y = self.__input_shape[1] // 2 + self.patch_size // 2
       mask[:, :uppleft_x, :] = 0
       mask[:, :, :uppleft_y] = 0
       mask[:, bottright_x:, :] = 0
       mask[:, :, bottright_y:] = 0
       return mask
    
    def set_transforms(self, translation_range, rotation_range, scale_range):
       self.__transforms = RandomAffine(rotation_range, translation_range, scale_range)

    def forward(self, img):
       patch, mask = self.transforms(self.__patch, self.__mask)
       inv_mask = torch.zeros_like(mask)
       inv_mask[mask == 0] = 1
       inv_mask = inv_mask.to(self.device)
       mask = mask.to(self.device)
       patch = patch.to(self.device)
       op = img * inv_mask + patch * mask
       return op

    

