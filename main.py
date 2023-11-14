import torch
import torchvision
import argparse
import tqdm
import os
import numpy as np
from torchvision.io import read_image, write_png
import random

import utils
import config as C
from attack import attack_folder
from attack_custom import attack_custom_folder, generate_attacked_dataset, unnormalize
import defenses

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="attack")
    parser.add_argument('--attack_type', type=str, default="custom")
    parser.add_argument('--defense_type', type=str, default="foundation")
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--lr', type=float, default=5.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_shape', type=str, default="square")
    return parser.parse_args()

def check_accuracy(image_folder, args, save_folder = None):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = utils.ImageDataset(
        image_folder,
        transform=transforms
    )
    data_classid2name = {
        v:k for k, v in dataset.class_to_idx.items()
    }
    
    model, model_id2name, _ = utils.get_model(args.model)
    

    acc_meter = utils.Accumulator()
    pbar = tqdm.tqdm(total = len(dataset))
    for img, label, img_name in dataset:

        prediction = model(img.unsqueeze(0)).squeeze(0).softmax(0)
        pred_cls = prediction.argmax().item()

        pred_name = model_id2name[pred_cls].replace(" ", "_")
        label_name = data_classid2name[label].replace(" ", "_")

        acc_meter.update(pred_name == label_name)

        pbar.update(1)
        pbar.set_description("Model {} Accuracy: {}".format(args.model, acc_meter))

        if save_folder is not None:
            cur_dir = os.path.join(save_folder, label_name)
            os.makedirs(cur_dir, exist_ok=True)
            write_png((255*unnormalize(img)).to(torch.uint8), os.path.join(cur_dir, img_name))

    return acc_meter    



def main():
    args = get_args()
    supported_models = [
        "resnet18", "inception", "resnet50", #"vit"
    ]
    
    if args.mode == "clean":
        img_folder = os.path.join(C.DATA_DIR, "val")
        for model in supported_models:
            args.model = model
            attacked_acc = check_accuracy(img_folder, args, save_folder="./data/clean")
            print("Model {} Clean Accuracy: {}".format(model, attacked_acc))

    elif args.mode == "attack":
        if args.attack_type == "im_patch":
            img_folder = os.path.join(C.DATA_DIR, "val")
            attack_folder_path = os.path.join(C.ATTACK_DIR, "val")
            attack_folder(img_folder, attack_folder_path, args)
            for model in supported_models:
                args.model = model
                attacked_acc = check_accuracy(attack_folder_path, args)
                print("Model {} im_patch Attacked Accuracy: {}".format(model, attacked_acc))
        elif args.attack_type == "custom":
            img_folder_train = os.path.join(C.DATA_DIR, "train")
            img_folder_val = os.path.join(C.DATA_DIR, "val")
            for model in supported_models:
                args.model = model
                patch_path=os.path.join("./patches/", args.patch_shape, model+".png")
                attack_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR, args.patch_shape, model, "val")
                plot_dir = os.path.join(C.ATTACK_PLOT_DIR, args.patch_shape, model)
                if not os.path.exists(patch_path):
                    attack_custom_folder(
                        img_folder_train, target=291, use_cuda=True, 
                        patch_path=patch_path, 
                        save_dir=attack_folder_path, plot_dir=plot_dir, args=args
                    )
                generate_attacked_dataset(img_folder_val, patch_path, attack_folder_path, args)
                attacked_acc = check_accuracy(attack_folder_path, args)
                print("Model {} custom-{} Attacked Accuracy: {}".format(model, args.patch_shape, attacked_acc))

    elif args.mode == "defense":
        img_folder = os.path.join(C.DATA_DIR, "val")
        for model in supported_models:
            args.model = model
            if args.attack_type == "im_patch":
                args.patch_shape = "square"
                attack_folder_path = os.path.join(C.ATTACK_DIR, "val")
            else:
                attack_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR, args.patch_shape, model, "val")

            if args.defense_type == "gradcam":
                defense_fn = defenses.gradcam_defense
                defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("gradcam"), args.attack_type, args.patch_shape, model, "val")
                defense_folder_check_path = defense_folder_path
                fn_args = (attack_folder_path, defense_folder_path, args)
            elif args.defense_type == "foundation":
                defense_fn = defenses.foundation_defense
                defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("foundation"), args.attack_type, args.patch_shape, "val")
                defense_folder_check_path = os.path.join(defense_folder_path, "image")
                fn_args = (attack_folder_path, defense_folder_path, args)
            elif args.defense_type == "mask_upper":
                defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("mask_upper"), args.attack_type, args.patch_shape, "val")
                defense_folder_check_path = defense_folder_path
                if args.attack_type == "im_patch":
                    defense_fn = attack_folder
                    fn_args = (img_folder, defense_folder_path, args, True)
                else:
                    defense_fn = generate_attacked_dataset
                    fn_args = (img_folder, "./patches/empty.png", defense_folder_path, args, True)
            elif args.defense_type == "inpaint":
                masked_folder_path = os.path.join(C.DEFENSE_DIRS.format("foundation"), args.attack_type, args.patch_shape, "val")
                defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("inpaint"), args.attack_type, args.patch_shape, "val")
                defense_folder_check_path = defense_folder_path
                defense_fn = defenses.inpaint_defense
                fn_args = (masked_folder_path, defense_folder_path, args)

            
            defense_fn(*fn_args)
            defended_acc = check_accuracy(defense_folder_check_path, args)
            print("Model {} Attacked {}-{} {} defense Accuracy: {}".format(model, args.attack_type, args.patch_shape, args.defense_type, defended_acc))





        
    # defenses.gradcam_defense(attack_folder_path, defense_folder_path, args)
    # defended_acc = check_accuracy(defense_folder_path, args)
    # print("Defense acc:", defended_acc)

    # 
    # generate_attacked_dataset(img_folder, patch_path="patch_lr{}_{}.png".format(args.lr, args.model), save_dir=attack_custom_folder_path)
    # attacked_acc = check_accuracy(attack_custom_folder_path, args)
    # print("Attacked accuracy = ", attacked_acc)

    # attacked_acc = check_accuracy(attack_custom_folder_path, args)
    # def_folder = os.path.join(C.DEFENSE_DIRS.format("gradcam_im_patch_swin"), "train")
    # defenses.gradcam_defense(attack_folder_path, def_folder, args)
    # acc = check_accuracy(def_folder, args)
    # print("ViT im-patch gradcam defense: ", acc)
    # defenses.gradcam_defense(attack_custom_folder_path, defense_custom_folder_path, args)
    # acc = check_accuracy(defense_custom_folder_path, args)
    # print("ViT custom-patch gradcam defense: ", acc)


    # defense_custom_folder_path = os.path.join(C.DEFENSE_DIRS.format("foundation_im_patch"), "train", "image")
    # # defenses.foundation_defense(attack_folder_path, defense_custom_folder_path, args)
    # for model in ["resnet18", "resnet50", "vit"]:
    #     args.model = model
    #     acc = check_accuracy(defense_custom_folder_path, args)
    #     print("Model {} Acc: {}".format(model, acc))

    # defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("mask_upper_imagenet_patch"), "train")
    # attack_folder(img_folder, defense_folder_path, args, zero_out=True)
    # for model in ["resnet18", "resnet50", "vit"]:
    #     args.model = model
    #     acc = check_accuracy(defense_folder_path, args)
    #     print("Model {} Acc: {}".format(model, acc))

    # defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("mask_upper"), "train")
    # generate_attacked_dataset(img_folder, patch_path="patch_empty.png".format(args.lr, args.model), save_dir=defense_folder_path)
    # for model in ["resnet18", "resnet50", "vit"]:
    #     args.model = model
    #     acc = check_accuracy(defense_folder_path, args)
    #     print("Model {} Acc: {}".format(model, acc))

if __name__ == "__main__":
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    main()