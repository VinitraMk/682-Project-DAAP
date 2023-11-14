import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from PIL import Image 

class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = []
        self.idx_to_class = {
            i:c for i, c in enumerate(os.listdir(root_dir)) if not ".DS_Store" in c
        }
        self.class_to_idx = {
            c:i for i, c in enumerate(os.listdir(root_dir)) if not ".DS_Store" in c
        }
        for class_name in sorted(os.listdir(root_dir)):
            if "DS_Store" in class_name: continue
            class_dir = os.path.join(root_dir, class_name)
            for img_name in sorted(os.listdir(class_dir)):
                if "DS_Store" in img_name: continue
                self.img_list.append({
                    "img_path": os.path.join(class_dir, img_name),
                    "class": self.class_to_idx[class_name],
                    "img_name": img_name,
                })


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        cur_dict = self.img_list[idx]
        img_path, class_id = cur_dict["img_path"], cur_dict["class"]

        image = Image.open(img_path).convert('RGB')
        # image = image / 255.0
        # print(image.shape, image.max(), image.min())

        if self.transform:
            image = self.transform(image)
        # else:
        #     image = torchvision.transforms.ToTensor()(image)
        # print(image.shape, image.max(), image.min())

        return image, class_id, cur_dict["img_name"]