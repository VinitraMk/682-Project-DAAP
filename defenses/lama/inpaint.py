import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import tqdm

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)



@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        mod=8,
        device="cuda",
        model=None,
        predict_config=None
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


def build_lama_model(        
        config_p: str,
        ckpt_p: str,
        device="cuda"
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    model.freeze()
    return model


@torch.no_grad()
def inpaint_img_with_builded_lama(
        model,
        img: np.ndarray,
        mask: np.ndarray,
        config_p=None,
        mod=8,
        device="cuda"
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch["inpainted"][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


def inpaint_single(model, img_path, mask_path, save_path, device, predict_config):

    img = load_img_to_array(img_path)
    mask = load_img_to_array(mask_path)
    if len(mask.shape) == 3:
        mask = mask.sum(-1)
    mask = mask > 0.5

    img_inpainted = inpaint_img_with_lama(
        img, mask, 
        device=device,
        model=model,
        predict_config=predict_config
    )
    save_array_to_img(img_inpainted, save_path)


def inpaint_defense(masked_img_folder, save_folder, args):

    lama_config = "/scratch/mchasmai/work/cs682/682-Project-DAAP/defenses/lama/config.yaml"
    lama_ckpt = "/scratch/mchasmai/work/cs682/Inpaint-Anything/pretrained_models/big-lama"

    device = "cuda" if torch.cuda.is_available() else "cpu"


    device = torch.device(device)
    predict_config = OmegaConf.load(lama_config)
    predict_config.model.path = lama_ckpt
    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)


    img_dir = os.path.join(masked_img_folder, "image")
    mask_dir = os.path.join(masked_img_folder, "mask")
    all_imgs = [1 for cl in os.listdir(img_dir) if "DS_Store" not in cl for _ in os.listdir(os.path.join(img_dir, cl))]
    defense_pbar = tqdm.tqdm(total = len(all_imgs), desc="Inpaint Defense")
    for class_name in os.listdir(img_dir):
        if "DS_Store" in class_name: continue
        class_dir = os.path.join(img_dir, class_name)
        os.makedirs(os.path.join(save_folder, class_name), exist_ok=True)
        for img_name in sorted(os.listdir(class_dir)):
            if "DS_Store" in img_name: continue
            img_path = os.path.join(img_dir, class_name, img_name)
            mask_path = os.path.join(mask_dir, class_name, img_name)
            save_path = os.path.join(save_folder, class_name, img_name)

            inpaint_single(model, img_path, mask_path, save_path, device, predict_config)
            defense_pbar.update(1)


if __name__ == "__main__":
    inpaint_defense(
        "/scratch/mchasmai/work/cs682/foundation/custom/circle/train",
        "/scratch/mchasmai/work/cs682/inpaints",
        None
    )