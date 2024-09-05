# to be run directly inside the yolov5 dataset directory.
# for eg: yolov5/data/attacked-images

import os

for split in ['train', 'val']:
    impath = os.path.join(os.getcwd(), f'{split}/images')
    all_files = [os.path.join(impath, fname) for fname in os.listdir(impath)]
    txt_path = os.path.join(os.getcwd(), f'{split}.txt')
    with open(txt_path, 'w+', encoding='utf-8') as fp:
        fp.writelines("\n".join(all_files))