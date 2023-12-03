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
import explanations

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="attack")
    parser.add_argument('--attack_type', type=str, default="custom")
    parser.add_argument('--attack_on', type=str, default="val")
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
    supported_shapes = [
        "square", "circle", "star"
    ]
    
    if args.mode == "clean":
        img_folder = os.path.join(C.DATA_DIR, "val")
        for model in supported_models:
            args.model = model
            attacked_acc = check_accuracy(img_folder, args, save_folder="./data/clean")
            print("Model {} Clean Accuracy: {}".format(model, attacked_acc))

    elif args.mode == "attack":

        if args.attack_type == "im_patch":
            img_folder = os.path.join(C.DATA_DIR, args.attack_on)
            if args.attack_on == "train":
                attack_folder_path = os.path.join(C.ATTACK_DIR, "train", "image")
                attack_mask_folder_path = os.path.join(C.ATTACK_DIR, "train", "mask")
            else:
                attack_folder_path = os.path.join(C.ATTACK_DIR, "val")
                attack_mask_folder_path = None
            attack_folder(img_folder, attack_folder_path, attack_mask_folder_path, args)
            if args.attack_on == "val":
                for model in supported_models:
                    args.model = model
                    attacked_acc = check_accuracy(attack_folder_path, args)
                    print("Model {} im_patch Attacked Accuracy: {}".format(model, attacked_acc))
        elif args.attack_type == "custom":
            img_folder = os.path.join(C.DATA_DIR, args.attack_on)
            img_folder_train = os.path.join(C.DATA_DIR, "train")
            for model in supported_models:
                args.model = model
                patch_path=os.path.join("./patches/", args.patch_shape, model+".png")
                if args.attack_on == "train":
                    attack_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR, args.patch_shape, model, "train", "image")
                    attack_mask_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR, args.patch_shape, model, "train", "mask")
                else:
                    attack_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR, args.patch_shape, model, "val")
                    attack_mask_folder_path = None
                plot_dir = os.path.join(C.ATTACK_PLOT_DIR, args.patch_shape, model)
                if not os.path.exists(patch_path):
                    attack_custom_folder(
                        img_folder_train, target=291, use_cuda=True, 
                        patch_path=patch_path, 
                        save_dir=attack_folder_path, save_mask_dir=attack_mask_folder_path,
                        plot_dir=plot_dir, args=args
                    )
                generate_attacked_dataset(img_folder, patch_path, attack_folder_path, attack_mask_folder_path, args)
                if args.attack_on == "val":
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
                defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("foundation"), args.attack_type, args.patch_shape, model, "val")
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
            elif args.defense_type == "sam-inpaint":
                masked_folder_path = os.path.join(C.DEFENSE_DIRS.format("foundation"), args.attack_type, args.patch_shape, model, "val")
                defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("inpaint"), args.attack_type, args.patch_shape, model, "val")
                defense_folder_check_path = defense_folder_path
                defense_fn = defenses.inpaint_defense
                fn_args = (masked_folder_path, defense_folder_path, args)
            elif args.defense_type == "unet-inpaint":
                if args.attack_type == "im_patch":
                    masked_folder_path = os.path.join(C.DEFENSE_DIRS.format("unet"), args.attack_type, "val")
                    defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("unet-inpaint"), args.attack_type, "val")
                else:
                    masked_folder_path = os.path.join(C.DEFENSE_DIRS.format("unet"), args.attack_type, args.patch_shape, model, "val")
                    defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("unet-inpaint"), args.attack_type, args.patch_shape, model, "val")
                defense_folder_check_path = defense_folder_path
                defense_fn = defenses.inpaint_defense
                fn_args = (masked_folder_path, defense_folder_path, args)
            elif args.defense_type == "unet":
                if args.attack_type == "im_patch":
                    checkpoint_dir = os.path.join(C.UNET_CHECKPOINT, args.attack_type)
                    train_dir = os.path.join(C.ATTACK_DIR, "train")
                    defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("unet"), args.attack_type, "val")
                else:
                    checkpoint_dir = os.path.join(C.UNET_CHECKPOINT, args.attack_type, args.patch_shape, model)
                    train_dir = os.path.join(C.ATTACK_CUSTOM_DIR, args.patch_shape, model, "train")
                    defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("unet"), args.attack_type, args.patch_shape, model, "val")
                if not os.path.exists(checkpoint_dir):
                    defenses.train_unet_patch(
                        train_dir, checkpoint_dir, 
                        use_cuda=True, args=args
                    )

                defense_folder_check_path = os.path.join(defense_folder_path, "image")
                defense_fn = defenses.unet_defense
                fn_args = (checkpoint_dir, attack_folder_path, defense_folder_path, args)

            
            defense_fn(*fn_args)
            defended_acc = check_accuracy(defense_folder_check_path, args)
            print("Model {} Attacked {}-{} {} defense Accuracy: {}".format(model, args.attack_type, args.patch_shape, args.defense_type, defended_acc))

    elif "cross" in args.mode:
        # cross model
        if "model" in args.mode:
            for shape in supported_shapes:
                for model1 in supported_models:
                    for model2 in supported_models:
                        args.model = model2
                        checkpoint_dir = os.path.join(C.UNET_CHECKPOINT, "custom", shape, model1)
                        attack_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR, shape, model2, "val")
                        defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("cross-unet"), "{}->{}".format(model1, model2), shape, "val")
                        
                        defense_folder_check_path = os.path.join(defense_folder_path, "image")
                    
                        defenses.unet_defense(checkpoint_dir, attack_folder_path, defense_folder_path, args)
                        defended_acc = check_accuracy(defense_folder_check_path, args)

                        print("Cross-model {} {} -> {}: Defended Acc: {}".format(shape, model1, model2, defended_acc))


        # cross shape
        if "shape" in args.mode:
            for model in supported_models:
                for shape1 in supported_shapes:
                    for shape2 in supported_shapes:
                        args.model = model
                        checkpoint_dir = os.path.join(C.UNET_CHECKPOINT, "custom", shape1, model)
                        attack_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR, shape2, model, "val")
                        defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("cross-unet"), "{}->{}".format(shape1, shape2), model, "val")
                        
                        defense_folder_check_path = os.path.join(defense_folder_path, "image")
                    
                        defenses.unet_defense(checkpoint_dir, attack_folder_path, defense_folder_path, args)
                        defended_acc = check_accuracy(defense_folder_check_path, args)

                        print("Cross-shape {} {} -> {}: Defended Acc: {}".format(model, shape1, shape2, defended_acc))


        # ImageNet -> custom square
        # custom square -> ImageNet
        if "attack" in args.mode:
            for model in supported_models:
                args.model = model
                for shape in supported_shapes:
                    checkpoint_dir = os.path.join(C.UNET_CHECKPOINT, "im_patch")
                    attack_folder_path = os.path.join(C.ATTACK_CUSTOM_DIR, shape, model, "val")
                    defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("cross-unet"), "img_path->custom", shape, model, "val")
                    defense_folder_check_path = os.path.join(defense_folder_path, "image")

                    defenses.unet_defense(checkpoint_dir, attack_folder_path, defense_folder_path, args)
                    defended_acc = check_accuracy(defense_folder_check_path, args)
                    print("Cross-attack {} {} img_path -> custom: Defended Acc: {}".format(shape, model, defended_acc))
                    
                    
                    
                    checkpoint_dir = os.path.join(C.UNET_CHECKPOINT, "custom", shape, model)
                    attack_folder_path = os.path.join(C.ATTACK_DIR, "val")
                    defense_folder_path = os.path.join(C.DEFENSE_DIRS.format("cross-unet"), "custom->img_path", shape, model, "val")
                    defense_folder_check_path = os.path.join(defense_folder_path, "image")
                    
                    defenses.unet_defense(checkpoint_dir, attack_folder_path, defense_folder_path, args)
                    defended_acc = check_accuracy(defense_folder_check_path, args)
                    print("Cross-attack {} {} custom -> img_path: Defended Acc: {}".format(shape, model, defended_acc))
                    



    elif args.mode == "explanation":
        # explanations.sam_corrupt_image(mult=0)
        # explanations.sam_corrupt_image(mult=0.2)
        # explanations.sam_corrupt_image(mult=0.5)
        # explanations.sam_corrupt_image(mult=0.8)
        # explanations.sam_corrupt_image(mult=1)

        # explanations.sam_smooth_image(smooth_num=1)
        # explanations.sam_smooth_image(smooth_num=2)
        # explanations.sam_smooth_image(smooth_num=3)
        # explanations.sam_smooth_image(smooth_num=4)
        # explanations.sam_smooth_image(smooth_num=5)

        # explanations.save_sam_pred(
        #     "./explanations/sample_images/inpaint.png",
        #     "./explanations/sample_images/inpaint_defend.png",
        # )
        # explanations.save_sam_pred(
        #     "./explanations/sample_images/defend.png",
        #     "./explanations/sample_images/defend_defend.png",
        # )

        # for prompt in [
        #     "adversarial", "patch", "counterfeit", "corruption", "unnatural", "out of distribution", "not dog"
        # ]:
        # for prompt in [
        #     # "stain", "edit", "fake", "novel", "spurious", "deceptive"
        #     "perturbation", "unknown"
        # ]:
        #     explanations.save_sam_pred(
        #         "./explanations/sample_images/attacked.png",
        #         "./explanations/sample_images/defend_prompt__{}__.png".format(prompt.replace(" ", "_")),
        #         text_prompt=prompt,
        #     )

        # explanations.save_sam_pred(
        #     "./explanations/sample_images/beach_astro.jpeg",
        #     "./explanations/sample_images/beach_astro_defend.png",
        #     "edit",
        # )
        # explanations.save_sam_pred(
        #     "./explanations/sample_images/beach_astro.jpeg",
        #     "./explanations/sample_images/beach_astro_defend__edit__.png",
        #     "edit",
        # )

        # explanations.save_sam_pred(
        #     "./explanations/sample_images/dalle_asstronaut.jpg",
        #     "./explanations/sample_images/dalle_asstronaut_defend.png",
        #     "edit",
        # )
        # explanations.save_sam_pred(
        #     "./explanations/sample_images/dalle_asstronaut.jpg",
        #     "./explanations/sample_images/dalle_asstronaut_defend__edit__.png",
        #     "edit",
        # )

        for model in supported_models:
            for shape in supported_shapes:
                explanations.unet_feature_invert(
                    model_save_dir = os.path.join(C.UNET_CHECKPOINT, "custom", shape, model),
                    save_dir = "./explanations/unet_feature_inversion/custom/{}/{}".format(shape, model),
                )

if __name__ == "__main__":
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    main()