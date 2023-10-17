# Adapted from https://github.com/pralab/ImageNet-Patch

import torch
import torchvision
from torchvision.transforms import functional as F


class MyRandomAffine(torchvision.transforms.RandomAffine):
    def forward(self, img, mask):
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                try:
                    fill = [float(fill)] * F.get_image_num_channels(img)
                except:
                    fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        try:
            img_size = F.get_image_size(img)
        except:
            img_size = F._get_image_size(img)
        ret = self.get_params(self.degrees, self.translate, self.scale,
                              self.shear, img_size)
        transf_img = F.affine(img, *ret, interpolation=self.interpolation,
                              fill=fill)
        transf_mask = F.affine(mask, *ret, interpolation=self.interpolation,
                               fill=fill)
        return transf_img, transf_mask

class ApplyPatch(torch.nn.Module):

    def __init__(self, patch, translation_range=(.2, .2), rotation_range=45,
                 scale_range=(0.5, 1), patch_size=50):
        super().__init__()
        self.patch_size = patch_size
        self.translation_range = translation_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range

        self._transforms = None
        self._patch = None
        self._input_shape = None
        self._mask = None

        self.set_transforms(translation_range, rotation_range, scale_range)
        self.set_patch(patch)

    @property
    def mask(self):
        return self._mask

    @property
    def transforms(self):
        return self._transforms

    def set_patch(self, patch):
        self._patch = patch
        self._input_shape = self._patch.shape
        self._mask = self._generate_mask()

    def _generate_mask(self):
        mask = torch.ones(self._input_shape)
        upp_l_x = self._input_shape[2] // 2 - self.patch_size // 2
        upp_l_y = self._input_shape[1] // 2 - self.patch_size // 2
        bott_r_x = self._input_shape[2] // 2 + self.patch_size // 2
        bott_r_y = self._input_shape[1] // 2 + self.patch_size // 2
        mask[:, :upp_l_x, :] = 0
        mask[:, :, :upp_l_y] = 0
        mask[:, bott_r_x:, :] = 0
        mask[:, :, bott_r_y:] = 0

        return mask

    def set_transforms(self, translation_range, rotation_range,
                       scale_range):
        self._transforms = MyRandomAffine(
            rotation_range, translation_range, scale_range)

    def forward(self, img):
        patch, mask = self.transforms(self._patch, self._mask)
        inv_mask = torch.zeros_like(mask)
        inv_mask[mask == 0] = 1
        return img * inv_mask + patch * mask