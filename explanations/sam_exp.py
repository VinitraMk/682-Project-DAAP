from defenses.foundation import LangSAM
from PIL import Image
import os
import cv2
import numpy as np

def save_sam_pred(img_path, save_path, text_prompt="adversarial patch"):
    model = LangSAM()
    img = cv2.imread(img_path)
    image_pil = Image.open(img_path).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    print(text_prompt, logits)
    try:
        mask = masks[0].numpy()[..., None]
    except IndexError:
        mask = np.zeros((224, 224, 1))

    cv2.imwrite(save_path, img * (1 - mask))


def sam_corrupt_image(sample_dir="./explanations/sample_images", mult=0.5):
    clean_image = cv2.imread(os.path.join(sample_dir, "clean.png"))
    mask_image = cv2.imread(os.path.join(sample_dir, "mask.png"))
    mask_image = mask_image > 100
    corrupt = clean_image * (1-mask_image) + mult * clean_image * mask_image
    cv2.imwrite(os.path.join(sample_dir, "corrupt_{}.png".format(mult)), corrupt)

    save_sam_pred(os.path.join(sample_dir, "corrupt_{}.png".format(mult)), os.path.join(sample_dir, "corrupt_{}_defend.png".format(mult)))


def sam_smooth_image(sample_dir="./explanations/sample_images", smooth_num=1):
    clean_image = cv2.imread(os.path.join(sample_dir, "attacked.png"))
    kernel = np.ones((5,5),np.float32)/25
    smooth = clean_image
    for _ in range(smooth_num):
        smooth = cv2.filter2D(smooth, -1, kernel)

    cv2.imwrite(os.path.join(sample_dir, "smooth_{}.png".format(smooth_num)), smooth)

    save_sam_pred(os.path.join(sample_dir, "smooth_{}.png".format(smooth_num)), os.path.join(sample_dir, "smooth_{}_defend.png".format(smooth_num)))