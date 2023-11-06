import os
from skimage import io

data_dir = 'data/imagenette2'

def cleanup_images(data_dir):
    data_abs = os.path.join(os.getcwd(), data_dir)
    c = 0
    n = 0
    for split in ['train', 'val']:
        split_path = os.path.join(data_abs, split)
        all_lbls = os.listdir(split_path)
        for lbl in all_lbls:
            lbl_path = os.path.join(split_path, lbl)
            lbl_dir = os.listdir(lbl_path)
            n += len(lbl_dir)
            for f in lbl_dir:
                img_path = os.path.join(lbl_path, f)
                img = io.imread(img_path)
                if len(img.shape) == 2:
                    os.remove(img_path)
                    c+=1
    print('Images deleted: ', c)
    print('Images left: ', (n-c))
    