import torch
import torchvision
import tqdm
import os
import numpy as np
import gzip
import pickle
from torchvision.io import read_image, write_png
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

import utils
import config as C

m = torch.Tensor([0.485, 0.456, 0.406])
s = torch.Tensor([0.229, 0.224, 0.225])
m = m.unsqueeze(1).unsqueeze(1)
s = s.unsqueeze(1).unsqueeze(1)
def unnormalize(img):    
    return img * s + m
    
IMG_SIZE = 224
PATCH_SIZE = 50

def get_translation(x, y, rot, w=IMG_SIZE, h=IMG_SIZE):
    if rot == 0:
        rotation_simple = np.array([[[1,0, -2*y/h],
                           [ 0,1, -2*x/w]]])
    elif rot == 1:
        rotation_simple = np.array([[[0,-1, -2*y/h],
                           [ 1,0, -2*x/w]]])
    elif rot == 2:
        rotation_simple = np.array([[[-1,0, -2*y/h],
                           [ 0,-1, -2*x/w]]])
    elif rot == 3:
        rotation_simple = np.array([[[0,1, -2*y/h],
                           [ -1,0, -2*x/w]]])
    return torch.Tensor(rotation_simple)


def apply_patch_random(img_batch, patch, mask, use_cuda):
    random_x, random_y = torch.randint(IMG_SIZE - PATCH_SIZE, (1, )), torch.randint(IMG_SIZE - PATCH_SIZE, (1, ))
    rot = torch.randint(4, (1, ))

    grid = F.affine_grid(get_translation(random_x.item(), random_y.item(), rot.item()), patch.size(), align_corners=False).type(torch.FloatTensor)
    mask_tr = F.grid_sample(mask, grid, mode="bilinear")
    patch_tr = F.grid_sample(patch, grid, mode="bilinear")
    # mask_tr = mask 
    # patch_tr = patch

    img_batch = img_batch * (1 - mask_tr) + patch_tr * mask_tr

    return img_batch, mask_tr

def get_mask(args):
    global PATCH_SIZE
    mask = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))
    if args.patch_shape == "square":
        mask[:, :, :PATCH_SIZE, :PATCH_SIZE] = 1
    elif args.patch_shape == "circle":
        PATCH_SIZE = 60
        for i_ in range(IMG_SIZE):
            for j_ in range(IMG_SIZE):
                if (i_ - PATCH_SIZE//2) ** 2 + (j_ - PATCH_SIZE//2) ** 2 < (PATCH_SIZE//2)**2:
                    mask[:, :, i_, j_] = 1

    elif args.patch_shape == "star":
        PATCH_SIZE = 80
        mask = cv2.imread("./patches/mask_star.png")
        mask = 1.0 * (mask > 127)
        mask = torch.Tensor(mask)
        mask = mask.transpose(1, 2).transpose(0, 1)
        mask = mask.unsqueeze(0)

    return mask



def attack_custom_folder(raw_img_folder, target, use_cuda, patch_path, save_dir, save_mask_dir, plot_dir, args):
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    # target = 291
    batch_size = args.batch_size
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = torchvision.datasets.ImageFolder(
        raw_img_folder,
        transform=transforms
    )
    data_classid2name = {
        v:k for k, v in dataset.class_to_idx.items()
    }

    np.random.seed(1)

    # class sampledDataset(torch.utils.data.Dataset):
    #     def __init__(self, dataset, rate=10):
    #         self.dataset = dataset
    #         self.rate = rate
    #     def __len__(self):
    #         return len(self.dataset)//self.rate
    #     def __getitem__(self, i):
    #         return self.dataset[i * self.rate]
    # dataset = sampledDataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model, model_id2name, _ = utils.get_model(args.model)
    if use_cuda:
        model = model.cuda()
    model.eval()

    # Initialise patch
    mask = get_mask(args)

    patch_img = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))
    patch_img[:, :, :PATCH_SIZE, :PATCH_SIZE] = torch.rand((3, PATCH_SIZE, PATCH_SIZE))
    patch_img.requires_grad_()
    # if use_cuda:
    #     patch_img = patch_img.cuda()
    #     mask = mask.cuda()
    patch_img = torch.nn.Parameter(patch_img)
    patch_img.requires_grad_()
    patch_optimizer = torch.optim.SGD([patch_img], lr = args.lr, momentum=0.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        patch_optimizer,
        milestones=[20, 30, 40], gamma=0.5
    )

    model_name2id = {
        v:k for k, v in enumerate(model_id2name)
    }
    target_name = C.CLASS_ID2NAME[target]
    target_name = target_name.replace("_", " ")
    target = model_name2id[target_name]
    adv_label = target * torch.ones(batch_size)
    adv_label = adv_label.to(torch.long)
    if use_cuda:
        adv_label = adv_label.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch2loss = []
    epoch2acc = []
    epoch2acc_adv = []
    num_epochs = 50
    pbar_epoch = tqdm.tqdm(total = num_epochs)
    relu_fn = torch.nn.ReLU(inplace=True)
    
    img_max = torch.Tensor([2.2489, 2.4286, 2.6400])
    img_min = torch.Tensor([-2.1179, -2.0357, -1.8044])
    img_max = img_max[None, :, None, None]
    img_min = img_min[None, :, None, None]

    for epoch in range(num_epochs):
        loss_accum = []
        pbar_batch = tqdm.tqdm(total = len(dataloader), leave=False)
        acc_meter = utils.Accumulator()
        adv_acc_meter = utils.Accumulator()
        for img_batch, labels in dataloader:

            patch_optimizer.zero_grad()

            # img_max, img_min = 0.8*img_batch.data.max(), 0.8*img_batch.data.min()
            # print(patch_img.max(), patch_img.min())
            
            img_batch, _ = apply_patch_random(img_batch, patch_img, mask, use_cuda)

            if use_cuda:
                img_batch = img_batch.cuda()

            output = model(img_batch)
            loss = loss_fn(output, adv_label[:img_batch.shape[0]]) 
            patch_only = (patch_img * mask)[:, :, :PATCH_SIZE, :PATCH_SIZE]
            pos_delta = relu_fn(patch_only - img_max)
            neg_delta = relu_fn(img_min - patch_only)
            # loss += 100 * (
            #     torch.nn.L1Loss()(pos_delta, torch.zeros_like(pos_delta)) + \
            #     torch.nn.L1Loss()(neg_delta, torch.zeros_like(neg_delta))
            # )
            loss += 100 * (
                torch.nn.MSELoss()(pos_delta, torch.zeros_like(pos_delta)) + \
                torch.nn.MSELoss()(neg_delta, torch.zeros_like(neg_delta))
            )
            # loss += torch.nn.MSELoss()(patch_only, torch.zeros_like(patch_only))

            pred_cls = output.argmax(dim=-1)
            pred_name = [model_id2name[p.item()].replace(" ", "_") for p in pred_cls]
            label_name = [data_classid2name[l.item()].replace(" ", "_") for l in labels]
            for p, l in zip(pred_name, label_name):
                acc_meter.update(p==l)
            for p, l in zip(pred_name, label_name):
                adv_acc_meter.update(p==target_name)


            loss.backward()
            patch_optimizer.step()

            with torch.no_grad():
                patch_img[:] = patch_img.clamp(img_min, img_max)
                # patch_img = relu_fn(patch_img - img_min) + img_min
                # patch_img = img_max - relu_fn(img_max - patch_img)

            pbar_batch.update(1)
            pbar_batch.set_description("Loss: {:.4f} | Model Acc {} Adv Acc {} | Patch min {:.3f} Max {:.3f}".format(
                loss.item(), str(acc_meter), str(adv_acc_meter),
                patch_img.min(), patch_img.max()
            ))
            loss_accum.append(loss.item())
        lr_scheduler.step()
        save_patch = 255*unnormalize(patch_img.data).squeeze(0)
        save_patch = mask.squeeze(0) * save_patch
        # print(save_patch.max(), save_patch.min())
        write_png(save_patch.to(torch.uint8), patch_path)
        # torch.save(save_patch.to(torch.uint8), "patch_banana_lr{}_{}.pt".format(args.lr, args.model))
        avg_loss = sum(loss_accum) / len(loss_accum)
        epoch2loss.append(avg_loss)
        epoch2acc.append(acc_meter.get_avg())
        epoch2acc_adv.append(adv_acc_meter.get_avg())
        pbar_epoch.update(1)
        pbar_epoch.set_description("Loss: {:.4f} | Model Acc: {} Adv Acc: {}".format(avg_loss, str(acc_meter), str(adv_acc_meter)))
        plt.plot(epoch2loss)
        plt.savefig(os.path.join(plot_dir, "attack_training_loss.png"), dpi=300)
        plt.close()
        plt.plot(epoch2acc, label="correct accuracy")
        plt.plot(epoch2acc_adv, label="adversarial accuracy")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "attack_training_acc.png"), dpi=300)
        plt.close()
    
    generate_attacked_dataset(raw_img_folder, patch_path, save_dir, save_mask_dir, args)

def generate_attacked_dataset(raw_img_folder, patch_path, save_dir, save_mask_dir, args, zero_out=False):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = utils.ImageDataset(
        raw_img_folder,
        transform=transforms
    )
    data_classid2name = {
        v:k for k, v in dataset.class_to_idx.items()
    }

    patch_img = read_image(patch_path)
    patch_img = patch_img[:3, ...]
    patch_img = patch_img/255.0
    patch_img = (patch_img - m)/ s
    patch_img = patch_img.unsqueeze(0)
    if zero_out:
        patch_img = -(m/s) * torch.ones_like(patch_img)

    mask = get_mask(args)

    count = 0
    for img, label, img_name in tqdm.tqdm(dataset, desc="Attacking"):
        
        label = C.CLASS_NAME2ID[data_classid2name[label]]

        img_batch = img.unsqueeze(0)
        img_batch, mask_tr = apply_patch_random(img_batch, patch_img, mask, use_cuda=False)
        img = img_batch.squeeze(0)
        mask_tr = mask_tr.squeeze(0)

        img = 255*unnormalize(img)

        os.makedirs(os.path.join(save_dir, str(C.CLASS_ID2NAME[label])), exist_ok=True)
        write_png(img.to(torch.uint8), os.path.join(save_dir, str(C.CLASS_ID2NAME[label]), img_name))

        if save_mask_dir is not None:
            os.makedirs(os.path.join(save_mask_dir, str(C.CLASS_ID2NAME[label])), exist_ok=True)
            write_png(255*mask_tr.to(torch.uint8), os.path.join(save_mask_dir, str(C.CLASS_ID2NAME[label]), img_name))

    
