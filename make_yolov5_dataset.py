import os
import pandas as pd
import shutil

def convert_to_yolov5_format(xmin, ymin, xmax, ymax, size = 224):
    boxw = xmax - xmin
    boxh = ymax - ymin
    xc = xmin + xmax / 2.0
    yc = ymin + ymax / 2.0
    xc/=size
    yc/=size
    boxw/=size
    boxh/=size
    return xc, yc, boxw, boxh

def make_labels(src_df, all_files, fnspath, label = 1):
    Np = 0
    Nn = 0
    for fn in all_files:
        if label == 1:
            row_val = src_df[src_df['Filename'] == f'{folder}/{fn}']
            src_fn = row_val['Filename'].values[0]
            fldnameidx = src_fn.index('/') + 1
            imgname = src_fn[fldnameidx:]
            xmin = row_val['xmin'].values[0]
            ymin = row_val['ymin'].values[0]
            xmax = row_val['xmax'].values[0]
            ymax = row_val['ymax'].values[0]
            xc, yc, w, h = convert_to_yolov5_format(xmin, ymin, xmax, ymax)
            fnstr = f"{label} {xc} {yc} {w} {h}"
            Np+=1
            dimgpath = os.path.join(os.getcwd(), f'yolov5-attacked-images/{split}/images/{imgname}')
            dlblpath = os.path.join(os.getcwd(), f'yolov5-attacked-images/{split}/labels/{imgname.replace("JPEG", "txt")}')
            srcimgpath = os.path.join(fnspath, f'{imgname}')
            shutil.copyfile(srcimgpath, dimgpath)
        else:
            fnstr = "0 0.0 0.0 0.0 0.0"
            Nn+=1
            dimgpath = os.path.join(os.getcwd(), f'yolov5-attacked-images/{split}/images/{fn}')
            dlblpath = os.path.join(os.getcwd(), f'yolov5-attacked-images/{split}/labels/{fn.replace("JPEG", "txt")}')
            srcimgpath = os.path.join(fnspath, f'{fn}')
            shutil.copyfile(srcimgpath, dimgpath)
            dlblpath = os.path.join(os.getcwd(), f'yolov5-attacked-images/{split}/labels/{fn.replace("JPEG", "txt")}')
            #print('neg example', srcimgpath, dimgpath, dlblpath)
        
        with open(dlblpath, 'w+') as fp:
            fp.write(fnstr)
    return Np, Nn

#reset directory
shutil.rmtree('yolov5-attacked-images')

all_yolov5_paths = ['yolov5-attacked-images',
'yolov5-attacked-images/train', 'yolov5-attacked-images/train/images', 'yolov5-attacked-images/train/labels',
'yolov5-attacked-images/val', 'yolov5-attacked-images/val/images', 'yolov5-attacked-images/val/labels'
]
# make all directories
for p in all_yolov5_paths:
    fpath = os.path.join(os.getcwd(), p)
    if not(os.path.exists(fpath)):
        os.mkdir(fpath)

data_dirs = ['English_springer', 'French_horn', 'cassette_player', 'chain_saw', 'church', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'tench']

neg_src_dir = 'data/imagenette2'
pos_src_dir = 'attacked-images'
N = 0
Np = 0
Nn = 0
for split in ['train', 'val']:
    src_path = os.path.join(os.getcwd(), f'{pos_src_dir}/{split}/{split}_results.csv')
    src_df = pd.read_csv(src_path)
    for folder in data_dirs:
        fnspath = os.path.join(os.getcwd(), f'{pos_src_dir}/{split}/{folder}')
        nfnspath = os.path.join(os.getcwd(), f'{neg_src_dir}/{split}/{folder}')
        all_files = os.listdir(fnspath)
        neg_files = os.listdir(nfnspath)
        posN = int(0.9 * len(all_files))
        negN = len(all_files) - posN
        pos_files = all_files[:posN]
        neg_files = neg_files[-negN:]
        np, nn = make_labels(src_df, pos_files, fnspath)
        Np+=np
        Nn+=nn
        N+=(np + nn)
        np, nn = make_labels(src_df, neg_files, nfnspath, 0)
        Np+=np
        Nn+=nn
        N+=(np + nn)
            

print('No of images copied', N, Np, Nn)
