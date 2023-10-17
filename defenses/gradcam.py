import torch
import torchvision
import argparse
import tqdm
import os
import numpy as np
import gzip
import pickle
from torchvision.io import read_image, write_png
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

import utils
import config as C

def gradcam_defense(attacked_img_folder, save_folder, args, filter_mode="none"):
    dataset = torchvision.datasets.ImageFolder(
        attacked_img_folder
    )
    data_classid2name = {
        v:k for k, v in dataset.class_to_idx.items()
    }

    model, model_id2name, target_layers = utils.get_model(args.model)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    g_cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    def get_gradcam(img):
        # targets = [ClassifierOutputTarget(30)]
        img_tr = transforms(img).unsqueeze(0)
        grayscale_cam = g_cam(input_tensor=img_tr, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam


    count = 0
    to_tensor = torchvision.transforms.ToTensor()
    for img, label in tqdm.tqdm(dataset, desc="Defending"):

        if filter_mode == "gt":
            prediction = model(transforms(img).unsqueeze(0)).squeeze(0).softmax(0)
            pred_cls = prediction.argmax().item()

            pred_name = model_id2name[pred_cls].replace(" ", "_")
            label_name = data_classid2name[label].replace(" ", "_")

            if pred_name != label_name:
                gcam = get_gradcam(img)
                gcam = (gcam - gcam.min())/(gcam.max() - gcam.min())
                gcam = gcam[:, :, None]
                

                img = img * (1 - gcam)
                img = torch.Tensor(img).transpose(0, 2)
            else:
                img = 255*to_tensor(img)
        elif filter_mode == "none":
            gcam = get_gradcam(img)
            gcam = gcam[:, :, None]
            img = img * (gcam)
            img = torch.Tensor(img).transpose(0, 2)
        else:
            gcam = get_gradcam(img)
            gcam = (gcam - gcam.min())/(gcam.max() - gcam.min())
            gcam = gcam[:, :, None]
            img_ = np.uint8(img * (1 - gcam))
            img_ = Image.fromarray(img_)
            gcam2 = get_gradcam(img_)
            gcam2 = (gcam2 - gcam2.min())/(gcam2.max() - gcam2.min())
            if gcam.std() > gcam2.std():
                img = img * (1 - gcam)
                img = torch.Tensor(img).transpose(0, 2)
            else:
                img = 255*to_tensor(img)
        os.makedirs(os.path.join(save_folder, str(data_classid2name[label])), exist_ok=True)
        write_png(img.to(torch.uint8), os.path.join(save_folder, str(data_classid2name[label]), "{}.jpg".format(count)))

        count += 1
