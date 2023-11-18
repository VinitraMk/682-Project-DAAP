### UNet model taken from https://github.com/TripleCoenzyme/ResNet50-Unet

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import tqdm
from PIL import Image 
import matplotlib.pyplot as plt
from torchvision.io import read_image, write_png

import utils


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
    
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode=='deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels = out_channels,

                                                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode=='pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x=self.conv(x)
        if self.BN_enable:
            x=self.norm1(x)
        x=self.relu1(x)
        x=self.upsample(x)
        if self.BN_enable:
            x=self.norm2(x)
        x=self.relu2(x)
        return x

class ResnetUnet(nn.Module):
    def __init__(self, BN_enable=True, resnet_pretrain=False):
        super().__init__()
        self.BN_enable = BN_enable
        resnet = models.resnet50(pretrained=resnet_pretrain)
        filters=[64,256,512,1024,2048]
        self.firstconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # decoder
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3], BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3]+filters[2], mid_channels=filters[2]*4, out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], mid_channels=filters[1]*4, out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1]+filters[0], mid_channels=filters[0]*4, out_channels=filters[0], BN_enable=self.BN_enable)
        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), 
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                nn.Sigmoid()
                )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1), 
                nn.Sigmoid()
                )

    def forward(self,x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        center = self.center(e3)

        d2 = self.decoder1(torch.cat([center,e2],dim=1))
        d3 = self.decoder2(torch.cat([d2,e1], dim=1))
        d4 = self.decoder3(torch.cat([d3,x], dim=1))

        return self.final(d4)

class SegDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "image")
        self.mask_dir = os.path.join(root_dir, "mask")
        self.transform = transform
        self.img_list = []
        self.idx_to_class = {
            i:c for i, c in enumerate(os.listdir(root_dir)) if not ".DS_Store" in c
        }
        self.class_to_idx = {
            c:i for i, c in enumerate(os.listdir(root_dir)) if not ".DS_Store" in c
        }
        for class_name in sorted(os.listdir(self.img_dir)):
            if "DS_Store" in class_name: continue
            class_dir = os.path.join(self.img_dir, class_name)
            for img_name in sorted(os.listdir(class_dir)):
                if "DS_Store" in img_name: continue
                self.img_list.append({
                    "img_path": os.path.join(class_dir, img_name),
                    "mask_path": os.path.join(self.mask_dir, class_name, img_name),
                    "img_name": img_name,
                })


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        cur_dict = self.img_list[idx]
        img_path, mask_path = cur_dict["img_path"], cur_dict["mask_path"]

        image = Image.open(img_path).convert('RGB')
        # image = image / 255.0
        # print(image.shape, image.max(), image.min())

        if self.transform:
            image = self.transform(image)

        mask = Image.open(mask_path).convert('RGB')
        mask = torchvision.transforms.ToTensor()(mask)
        mask = mask[0:1, ...]
        mask = mask > 0.5
        mask = mask.to(torch.float)


        return image, mask, cur_dict["img_name"]

def train_unet_patch(train_attacked_folder, model_save_dir, use_cuda, args):
    batch_size = args.batch_size
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = SegDataset(
        train_attacked_folder,
        transform=transforms
    )
    data_classid2name = {
        v:k for k, v in dataset.class_to_idx.items()
    }
    np.random.seed(1)



    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    unet = ResnetUnet()
    unet.train()
    if use_cuda:
        unet = unet.cuda()

    optimizer = torch.optim.SGD(unet.parameters(), lr = args.lr, momentum=0.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 30, 40], gamma=0.5
    )
    num_epochs = 10
    pbar_epoch = tqdm.tqdm(total = num_epochs)
    epoch2loss = []

    for epoch in range(num_epochs):
        loss_accum = []
        pbar_batch = tqdm.tqdm(total = len(dataloader), leave=False)
        for img_batch, labels, _ in dataloader:
            optimizer.zero_grad()

            if use_cuda:
                img_batch = img_batch.cuda()
                labels = labels.cuda()

            predictions = unet(img_batch)

            loss = torch.nn.BCELoss()(
                predictions, labels
            )
            loss.backward()
            optimizer.step()

            pbar_batch.update(1)
            pbar_batch.set_description("Loss: {:.4f}".format(loss.item()))
            loss_accum.append(loss.item())

        os.makedirs(model_save_dir, exist_ok=True)
        lr_scheduler.step()
        avg_loss = sum(loss_accum) / len(loss_accum)
        epoch2loss.append(avg_loss)

        pbar_epoch.update(1)
        pbar_epoch.set_description("Loss: {:.4f}".format(avg_loss))

        plt.plot(epoch2loss)
        plt.savefig(os.path.join(model_save_dir, "training_loss.png"), dpi=300)
        plt.close()
        
        torch.save(unet.state_dict(), os.path.join(model_save_dir, "model.ckpt"))


def unet_defense(model_save_dir, attacked_img_folder, save_folder, args):
    unet = ResnetUnet()
    unet.load_state_dict(
        torch.load(os.path.join(model_save_dir, "model.ckpt"))
    )
    unet.eval()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset = utils.ImageDataset(
        attacked_img_folder,
        transform=transforms
    )
    data_classid2name = {
        v:k for k, v in dataset.class_to_idx.items()
    }
    to_tensor = torchvision.transforms.ToTensor()
    for img, label, img_name in tqdm.tqdm(dataset, desc="UNet - Defending"):

        img_save_path = os.path.join(save_folder, "image", str(data_classid2name[label]), img_name)
        mask_save_path = os.path.join(save_folder, "mask", str(data_classid2name[label]), img_name)

        if os.path.exists(img_save_path) and os.path.exists(mask_save_path):
            continue

        with torch.no_grad():
            patch_pred = unet(normalize(img).unsqueeze(0))
            patch_pred = patch_pred.squeeze(0)

        img_def = (1 - patch_pred) * img
        # print(patch_pred.shape, img.shape, img_def.shape)

        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
        write_png((255*img_def).to(torch.uint8), img_save_path)

        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        write_png((255*(patch_pred > 0.5)).to(torch.uint8), mask_save_path)
