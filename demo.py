import torchvision
from torchvision.io import read_image, write_png
from attack_custom import *
import utils
import numpy as np
from main import get_args
import tqdm
import defenses
import cv2

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def apply_patch_xy(img, patch, mask, x, y):
    grid = F.affine_grid(get_translation(x, y, 0), patch.size(), align_corners=False).type(torch.FloatTensor)
    mask_tr = F.grid_sample(mask, grid, mode="bilinear")
    patch_tr = F.grid_sample(patch, grid, mode="bilinear")


    img = img * (1 - mask_tr) + patch_tr * mask_tr

    return img, mask_tr

def generate_attacked_images(img_path, patch_path, save_dir, args):

    os.makedirs(save_dir, exist_ok=True)

    patch_img = read_image(patch_path)
    patch_img = patch_img[:3, ...]
    patch_img = patch_img/255.0
    # patch_img = (patch_img - m)/ s
    patch_img = patch_img.unsqueeze(0)

    img = read_image(img_path)/255.0
    img = img.unsqueeze(0)

    mask = get_mask(args)

    model, model_id2name, _ = utils.get_model(args.model)

    np.random.seed(123)

    x, y = np.random.randint(IMG_SIZE - PATCH_SIZE), np.random.randint(IMG_SIZE - PATCH_SIZE)
    for count in tqdm.tqdm(range(500)):

        img_cur, mask_cur = apply_patch_xy(img, patch_img, mask, x, y)

        img_cur = 255 * img_cur.squeeze(0)
        write_png(img_cur.to(torch.uint8), os.path.join(save_dir, "{}.jpg".format(count)))


        direc = np.random.randint(4)
        step_size = np.random.randint(7, 20)

        step = {
            0: (0, 1),
            1: (1, 0),
            2: (0, -1),
            3: (-1, 0),
        }
        dx, dy = step[direc]

        x += dx * step_size
        y += dy * step_size

        x = max(x, 0)
        y = max(y, 0)
        x = min(x, IMG_SIZE - PATCH_SIZE)
        y = min(y, IMG_SIZE - PATCH_SIZE)


def save_best_accs(img_dir, save_dir, args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = utils.ImageDataset(
        img_dir,
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


        cur_dir = os.path.join(save_dir, label_name)
        os.makedirs(cur_dir, exist_ok=True)
        scores, clss = torch.topk(prediction, 5)
        x = [model_id2name[c_id] for c_id in clss]
        colors = ["red"] * 5
        for i in range(len(x)):
            if x[i].replace(" ", "_") ==  label_name:
                colors[i] = "green"
            x[i].replace("-", "")
            if len(x[i]) > 20 and len(x[i].split(" ")) > 2:
                x[i] = " ".join(x[i].split(" ")[:2])

        my_dpi = 300
        # plt.figure(figsize=(IMG_SIZE/my_dpi, IMG_SIZE/my_dpi), dpi=my_dpi)
        plt.figure(figsize=(5, 5), dpi=my_dpi)
        plt.bar(x, scores.detach(), color=colors)
        plt.ylim(0, 1)
        plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        # plt.xticks(fontsize=8)#, rotation=5)
        plt.tight_layout()
        plt.savefig(os.path.join(cur_dir, img_name), dpi=my_dpi)
        plt.close()

        
        acc_meter.update(pred_name == label_name)

        pbar.update(1)
        pbar.set_description("Model {} Accuracy: {}".format(args.model, acc_meter))

    return acc_meter

def make_gif(img_dir, save_path):
    import imageio.v2 as imageio
    with imageio.get_writer(save_path, mode='I', duration=0.1, loop=1000) as writer:
        for idx in tqdm.tqdm(range(100)):
            image = imageio.imread(os.path.join(img_dir, "{}.jpg".format(idx)))
            writer.append_data(image)

def concat_all(input_dirs, save_dir, padding, label):
    num_images = len(input_dirs)
    num_rows = len(input_dirs[0])

    final_img_w = num_images * IMG_SIZE + (num_images + 1) * padding
    final_img_h = num_rows * IMG_SIZE + (num_rows + 1) * padding

    os.makedirs(save_dir, exist_ok=True)

    for img_name in tqdm.tqdm(os.listdir(os.path.join(input_dirs[0][0], label))):
        combined_image = 255 * np.ones((final_img_h, final_img_w, 3))
        for col, method_dir in enumerate(input_dirs):
            cur_img_dir, cur_pred_dir = method_dir

            img = cv2.imread(os.path.join(cur_img_dir, label, img_name))
            combined_image[
                padding:padding+IMG_SIZE,
                padding + col * (IMG_SIZE + padding): (col + 1) * (IMG_SIZE + padding),
                :
            ] = img

            pred_img = cv2.imread(os.path.join(cur_pred_dir, label, img_name))
            pred_img = cv2.resize(pred_img, (IMG_SIZE, IMG_SIZE))
            combined_image[
                2 * padding + IMG_SIZE: 2 * (padding + IMG_SIZE),
                padding + col * (IMG_SIZE + padding): (col + 1) * (IMG_SIZE + padding),
                :
            ] = pred_img
        cv2.imwrite(os.path.join(save_dir, img_name), combined_image)



 


if __name__ == "__main__":
    args = get_args()
    label = "English_springer"
    demo_root = "./data/demo1"
    img_path = "./data/clean/English_springer/n02102040_460.JPEG"
    
    patch_path = "/scratch/mchasmai/work/cs682/682-Project-DAAP/patches/circle/inception.png"
    args.patch_shape = "circle"
    PATCH_SIZE = 60 

    args.model = "inception"


    attack_dir = os.path.join(demo_root, "attacked")
    # generate_attacked_images(
    #     img_path,
    #     patch_path,
    #     os.path.join(attack_dir, label),
    #     args
    # )
    # save_best_accs(attack_dir, os.path.join(demo_root, "preds_attacked"), args)

    # defenses.gradcam_defense(
    #     attack_dir, 
    #     os.path.join(demo_root, "gradcam"),
    #     args
    # )
    # save_best_accs(os.path.join(demo_root, "gradcam"), os.path.join(demo_root, "preds_gradcam"), args)

    # defenses.foundation_defense(
    #     attack_dir, 
    #     os.path.join(demo_root, "sam"),
    #     args
    # )
    # save_best_accs(os.path.join(demo_root, "sam", "image"), os.path.join(demo_root, "preds_sam"), args)

    # defenses.inpaint_defense(
    #     os.path.join(demo_root, "sam"), 
    #     os.path.join(demo_root, "inpaint"),
    #     args
    # )
    # save_best_accs(os.path.join(demo_root, "inpaint"), os.path.join(demo_root, "preds_inpaint"), args)

    # input_dirs = [
    #     [os.path.join(demo_root, "attacked"), os.path.join(demo_root, "preds_attacked")],
    #     [os.path.join(demo_root, "gradcam"), os.path.join(demo_root, "preds_gradcam")],
    #     [os.path.join(demo_root, "sam", "image"), os.path.join(demo_root, "preds_sam")],
    #     [os.path.join(demo_root, "inpaint"), os.path.join(demo_root, "preds_inpaint")],
    # ]
    # concat_all(
    #     input_dirs, 
    #     save_dir=os.path.join(demo_root, "combined_img"),
    #     padding=10,
    #     label=label
    # )

    make_gif(
        img_dir=os.path.join(demo_root, "combined_img"), 
        save_path=os.path.join(demo_root, "demo.gif")
    )