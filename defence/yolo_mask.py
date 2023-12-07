import os
import subprocess
import pandas as pd
from torchvision.io import read_image
from torchvision.utils import save_image
import shutil
import torch

class YOLOMask:
    yolo_dir = 'yolov5'
    attacked_imgs_dir = 'data/attacked-images'
    test_dir = 'yolov5/data/yolov5-attacked-images/val/images'
    src_dir = ''
    weights_path = 'runs/train/yolo-patch-detection-sm5/weights/best.pt'
    script_train = 'yolov5/train.py'
    script_detect = 'yolov5/detect.py'
    results_path = 'yolov5/runs/detect/exp2/labels'
    columns = ['filename', 'patch-class', 'true-label', 'clean-prediction', 'attacked-prediction', 'defence-prediction', 'xmin', 'ymin', 'xmax', 'ymax']
    output_dir = 'output/attacked-imgs-bbox-preds.csv'
    mask_dir = 'data/yolo-mask-images'

    def __init__(self):
        self.root_dir = os.getcwd()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def __yolo2voc(self, fpath):
        iw = 224
        ih = 224
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            label, xc, yc, w, h = lines[0].rstrip().split(" ")
            xc = float(xc)
            yc = float(yc)
            w = float(w)
            h = float(h)
            xp = xc * iw
            yp = yc * ih
            hw = w * iw / 2
            hh = h * ih / 2
            xmin = int(xp - hw)
            ymin = int(yp - hh)
            xmax = int(xp + hw)
            ymax = int(xp + hh)
        return label, xmin, ymin, xmax, ymax

    def get_rel2root(self, spath):
        return os.path.join(self.root_dir, spath)

    def __make_yolomasks(self, sfn, mfn, mode, xmin, ymin, xmax, ymax):
        ippath = os.path.join(os.getcwd(), f'{self.attacked_imgs_dir}/{mode}/{sfn}')
        dpath = os.path.join(os.getcwd(), f'{self.mask_dir}/{mfn.replace(".txt", ".JPEG")}')
        img = read_image(ippath)
        img[:, ymin-1:ymax+1, xmin-1:xmax+1] = 0.0
        save_image(img/255, dpath)
        return img/255
        
    def __initialize_mask_dirs(self):
        fpath = os.path.join(os.getcwd(), self.mask_dir)
        if os.path.exists(fpath):
            shutil.rmtree(fpath)
        os.mkdir(fpath)
    
    def __load_yolo(self, weight_dir = 'yolov5/runs/train/yolo-patch-detection-sm-final/weights/best.pt'):
        print(os.getcwd())
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom',
        path='yolov5/runs/train/yolo-patch-detection-sm-final/weights/best.pt', force_reload=True)

    def read_test_results(self, test_dir, mode, model):
        model.eval()
        self.results_path = test_dir
        self.__initialize_mask_dirs()
        results_path = self.get_rel2root(self.results_path)
        print('experiment to be evaluated', results_path)
        all_lbls = os.listdir(results_path)
        batch_mask_imgs = torch.empty((1, 3, 224, 224))
        srccsv = f'{mode}/{mode}_results.csv'
        val_df = pd.read_csv(os.path.join(self.attacked_imgs_dir, srccsv))
        rows = []
        for fname in all_lbls:
            lbl, xmin, ymin, xmax, ymax = self.__yolo2voc(os.path.join(results_path, fname))
            row = val_df[val_df['Filename'].str.contains(fname.replace(".txt", ".JPEG"))]
            vals = row['Filename'].values
            if (len(vals) > 0):
                srcfn = vals[0]
                tl = row['true-label'].values[0]
                cp = row['clean-prediction'].values[0]
                ap = row['attacked-prediction'].values[0]
                mimg = self.__make_yolomasks(srcfn, fname, mode, xmin, ymin, xmax, ymax).double()
                dp = torch.argmax(model(mimg.unsqueeze(0).to(self.device)), dim = 1)
                #batch_mask_imgs = torch.concat([batch_mask_imgs, mimg.unsqueeze(0)])
                predrow = [fname.replace(".txt", ".JPEG"), lbl, tl, cp, ap, dp, xmin, ymin, xmax, ymax]
                rows.append(predrow)

        df = pd.DataFrame(rows, columns = self.columns)
        cspath = os.path.join(self.mask_dir, 'attacked-imgs-bbox-preds.csv')
        df.to_csv(cspath)
        return df

    def run(self):
        print('start program YOLO mask')
        self.read_test_results()

if __name__ == "__main__":
    sign_indp = YOLOMask()
    sign_indp.run()
