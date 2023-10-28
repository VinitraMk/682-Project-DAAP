import torch
import torchvision
import argparse
import tqdm
import os
import numpy as np
from torchvision.io import read_image, write_png

import utils
import config as C
from attack import attack_folder
from attack_custom import attack_custom_folder, generate_attacked_dataset
import defenses

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--save_attacked', dest='save_attacked', action='store_true')
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()

def check_accuracy(image_folder, args):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = torchvision.datasets.ImageFolder(
        image_folder,
        transform=transforms
    )
    data_classid2name = {
        v:k for k, v in dataset.class_to_idx.items()
    }
    
    model, model_id2name, _ = utils.get_model(args.model)
    

    acc_meter = utils.Accumulator()
    pbar = tqdm.tqdm(total = len(dataset))
    for img, label in dataset:
        # print(img.shape, cl)
        prediction = model(img.unsqueeze(0)).squeeze(0).softmax(0)
        pred_cls = prediction.argmax().item()

        pred_name = model_id2name[pred_cls].replace(" ", "_")
        label_name = data_classid2name[label].replace(" ", "_")

        acc_meter.update(pred_name == label_name)

        pbar.update(1)
        pbar.set_description("Accuracy: {}".format(acc_meter))

    return acc_meter    



def main():
    args = get_args()
    print("Model: {}, lr: {}".format(args.model, args.lr))
    img_folder = os.path.join(C.DATA_DIR, "train")
    attack_folder_path = os.path.join(C.ATTACK_DIR, "train")
    defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("gradcam"), "train")
    attack_custom_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR + "_lr{}_{}".format(args.lr, args.model), "train")
    defense_custom_folder_path = os.path.join(C.DEFENSE_DIRS.format("gradcam") + "_lr{}_{}".format(args.lr, args.model), "train")
    
    # clean_acc = check_accuracy(img_folder, args)
    # if args.save_attacked:
        # attack_folder(img_folder, attack_folder_path, args)
    # attacked_acc = check_accuracy(attack_folder_path, args)
    # defenses.gradcam_defense(attack_folder_path, defense_folder_path, args)
    # defended_acc = check_accuracy(defense_folder_path, args)
    # print("Defense acc:", defended_acc)

    # attack_custom_folder(img_folder, 291, use_cuda=True, patch_path="patch_lr{}_{}.png".format(args.lr, args.model), save_dir=attack_custom_folder_path, args=args)
    # generate_attacked_dataset(img_folder, patch_path="patch_lr{}_{}.png".format(args.lr, args.model), save_dir=attack_custom_folder_path)
    # attacked_acc = check_accuracy(attack_custom_folder_path, args)
    # print("Attacked accuracy = ", attacked_acc)

    # attacked_acc = check_accuracy(attack_custom_folder_path, args)
    defenses.gradcam_defense(attack_custom_folder_path, defense_custom_folder_path, args)
    check_accuracy(defense_custom_folder_path, args)

main()