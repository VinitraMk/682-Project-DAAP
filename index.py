# starter code goes here
from utils import utils
from utils.cleanup import cleanup_images
from attack.patch_attack import PatchAttack
from models.resnet18 import Resnet18
from models.resnet50 import Resnet50
from models.inceptionv3 import InceptionV3
from defence.signature_indp import SignatureIndp
from defence.yolo_mask import YOLOMask


import os
import torch
from datautils.dataset import ImageNetDataset
from datautils.dataloader import get_dataloader
from torchvision import transforms as tvtransforms
from torch.utils.data import DataLoader
import pandas as pd
import gzip
import pickle
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import argparse
from torchvision.utils import save_image
import random
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision.utils import draw_bounding_boxes

class Index:
    data_dir = ''
    output_dir = ''
    train_labels_path = ''
    lbl_le_mapping = {}
    model = None
    model_name = 'resnet18'
    columns = ['Filename', 'true-label', 'clean-prediction', 'attacked-prediction', 'defence-prediction', 'xmin', 'ymin', 'xmax', 'ymax']

    def __init__(self, model_name):
        self.data_dir = 'data/imagenette2'
        self.patch_dir = 'data/attacked-images'
        self.output_dir = 'output'
        self.train_labels_path = self.data_dir + '/train_images.csv'
        self.val_labels_path = self.data_dir + '/val_images.csv'
        self.train_df = pd.DataFrame([], columns = self.columns)
        self.val_df = pd.DataFrame([], columns=self.columns)
        self.model_name = model_name
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def initialize_transforms(self, rescale_size = 256, crop_size = 224):
        self.normalizer = utils.get_normalizer()
        self.unnormalizer = utils.get_inv_normalizer()
        self.data_transforms = utils.get_imagenette_transforms(rescale_size, crop_size)

    def clear_outputdirs(self):
        print("Reset output directories")
        data_dirs = ['English_springer', 'French_horn', 'cassette_player', 'chain_saw', 'church', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'tench']
        target_dir = os.path.join(os.getcwd(), f"{self.patch_dir}/train")
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
        for folder in data_dirs:
            fpath = os.path.join(target_dir, folder)
            os.mkdir(fpath)
        target_dir = os.path.join(os.getcwd(), f"{self.patch_dir}/val")
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
        for folder in data_dirs:
            fpath = os.path.join(target_dir, folder)
            os.mkdir(fpath)

    def check_accuracy(self, df, test_len):
        clean_sum = (df['clean-prediction'] == df['true-label']).sum()
        attack_sum = (df['attacked-prediction'] == df['true-label']).sum()
        defence_sum = (df['defence-prediction'] == df['true-label']).sum()
        print('\nClean image accuracy: ', clean_sum / test_len)
        print('Attacked image accuracy: ', attack_sum / test_len)
        print('Defence image accuracy: ', defence_sum / test_len)

    def start_program(self, rescale_size = 256, crop_size = 224,
        operation = 'train-classifier', defence_type = 'sign-indp',
        epochs = 2, test_dir = 'yolov5/runs/detect/exp2/labels', subset_size = -1):
        # clean up gray scale images from the directory [temporary solution]
        print('program started for operation', operation)
        #cleanup_images(self.data_dir)
        #utils.make_csv(self.data_dir)
        self.initialize_transforms(rescale_size, crop_size)
        self.build_datasetloader(subset_size)
        self.initialize_model(self.model_name)
        if operation == 'train-classifier':
            self.retrain_classifier(epochs)
        elif operation == 'attack':
            self.clear_outputdirs()
            self.load_model()
            self.add_patches_to_imgs()
        elif operation == 'defence':
            self.load_model()
            self.start_defence(defence_type, test_dir) 

        
    def build_datasetloader(self, subset_size = -1):
        print('building datasets...')
        self.lbl_le_mapping = utils.make_csv(self.data_dir)

        batch_size = 64

        imagenette_train_dataset = ImageNetDataset(labels_path = self.train_labels_path,
                                                data_dir = os.path.join(self.data_dir, 'train'),
                                                transform=self.data_transforms,
                                                lbl_mapping=self.lbl_le_mapping)
        imagenette_val_dataset =  ImageNetDataset(labels_path = self.val_labels_path,
                                                data_dir = os.path.join(self.data_dir, 'val'),
                                                transform=self.data_transforms,
                                                lbl_mapping=self.lbl_le_mapping)
        if subset_size == -1:
            self.train_dataloader, self.val_dataloader, self.train_len, self.val_len = get_dataloader(imagenette_train_dataset, imagenette_val_dataset)
        else:
            self.train_dataloader, self.val_dataloader, self.train_len, self.val_len = get_dataloader(imagenette_train_dataset, imagenette_val_dataset, subset_size, True)
        print('\nLength of train dataset: ', self.train_len)
        print('Length of val dataset: ', self.val_len, '\n')

    def initialize_model(self, model_name = 'resnet18'):
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.loss = torch.tensor(0.0, requires_grad=True)

        self.model_name = model_name
        if model_name == 'resnet18':
            self.model = Resnet18()
        elif model_name == 'resnet50':
            self.model = Resnet50()
        elif model_name == 'inception':
            self.model = InceptionV3()

        self.model = self.model.double()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.001, momentum=0.9)
        self.model.to(self.device)
    
    def __draw_bounding_box(self, attacked_img_batch, box_coord):
        for i in range(attacked_img_batch.size()[0]):
            boxes = torch.tensor([[box_coord[i][0], box_coord[i][1], box_coord[i][2], box_coord[i][3]]])
            attacked_img_batch[i] = draw_bounding_boxes(image=(255 * attacked_img_batch[i]).to(torch.uint8),
                boxes=boxes, colors="red")
            attacked_img_batch[i] /= 255
        return attacked_img_batch

    def __save_attacked_images(self, imgs, filenames, mode = 'train'):
        unnormalized_imgs = self.unnormalizer(imgs)
        for i in range(len(unnormalized_imgs)):
            img = unnormalized_imgs[i].cpu()#.transpose(1, 2, 0)
            #test_dir = os.path.join(os.getcwd(), 'test-attacked-images')
            #save_image(img, os.path.join(test_dir, 'sample.JPEG'))
            output_dir = os.path.join(os.getcwd(), f'{self.patch_dir}/{mode}')
            save_image(img, os.path.join(output_dir, f"{filenames[i]}"))

    def __insert_patch_on_batch(self, img_batch, lbls, filenames, target_df, mode = 'train'):
        filepath = os.path.join(os.getcwd(), f'{self.patch_dir}/{mode}/{mode}_results.csv')
        with open(os.path.join(os.getcwd(), "data/class_id2name.json")) as f:
           target_to_classname = eval(f.read())

        with gzip.open(os.path.join(os.getcwd(), "data/imagenet_patch.gz"), "rb") as f:
             imagenet_patch = pickle.load(f)

        patches, targets, info = imagenet_patch
        patch_idx = torch.randint(0, len(patches), (1,))[0].item()
        patch = patches[patch_idx]

        apply_patch = PatchAttack(self.model, 'output', patch)

        patch_normalizer = Compose([apply_patch, self.normalizer])

        x_clean = self.normalizer(img_batch).to(self.device)
        x_attacked, bb_idx = apply_patch(img_batch)
        x_attacked = self.__draw_bounding_box(x_attacked, bb_idx)
        x_attacked = self.normalizer(x_attacked).to(self.device)
        self.__save_attacked_images(x_attacked, filenames, mode)
        op_clean = self.model(x_clean.double())
        preds_clean = torch.argmax(op_clean, dim=1, keepdims=True).cpu().numpy()
        op_attacked = self.model(x_attacked.double())
        preds_attacked =  torch.argmax(op_attacked, dim=1, keepdims=True).cpu().numpy()
        x_clean.cpu()
        x_attacked.cpu()
        defpreds = torch.ones((x_clean.size()[0],)) * -1
        pd_data = []
        for fn,y,cpred,apred,dpred,coords in zip(filenames, lbls, preds_clean, preds_attacked, defpreds, bb_idx):
            pd_data.append([fn, y.item(), cpred[0], apred[0], dpred.item(), coords[0].item(), coords[1].item(), coords[2].item(), coords[3].item()])
        pdx = pd.DataFrame(pd_data, columns=self.columns)
        target_df = pd.concat([target_df, pdx], ignore_index = True)
        target_df.to_csv(filepath)
        return target_df

    def __plot_loss_history(self, loss_history, filename):
        plt.plot(list(range(len(loss_history))), loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss history')
        plt.savefig(f'{filename}.png')
        plt.clf()

    def add_patches_to_imgs(self):
        print('\nAttack images')
        self.model.eval()
        for i_batch, sample_batch in enumerate(self.train_dataloader):
            print(f'\tAdding patches to train batch {i_batch}')
            self.train_df = self.__insert_patch_on_batch(sample_batch['image'],
            sample_batch['label'],
            sample_batch['filename'], self.train_df, 'train')

        for i_batch, sample_batch in enumerate(self.val_dataloader):
            print(f'\tAdding patches to val batch {i_batch}')
            self.val_df = self.__insert_patch_on_batch(sample_batch['image'],
            sample_batch['label'],
            sample_batch['filename'], self.val_df, 'val')

    def retrain_classifier(self, num_epochs = 2):
        print('Retraining classifier on 10 classes')
        output_dir = os.path.join(os.getcwd(), self.patch_dir)
        self.model = self.model.double()
        self.model.train()
        loss_history = []
        for i in range(num_epochs):
            print(f"Running epoch {i}")
            running_loss = 0.0
            for i_batch, sample_batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                imgs = self.normalizer(sample_batch['image']).to(self.device).double()
                lbls = sample_batch['label'].to(self.device)
                output = self.model(imgs)
                loss = self.loss_criterion(output, lbls)
                loss.backward()
                running_loss += loss.detach().item()
                self.optimizer.step()
                del imgs
                del lbls
            running_loss /= len(self.train_dataloader)
            loss_history.append(running_loss)
        
        self.__plot_loss_history(loss_history, 'train_loss_history')
        self.model.eval()
        running_loss = 0.0
        acc = 0.0
        for i_batch, sample_batch in enumerate(self.val_dataloader):
            imgs = self.normalizer(sample_batch['image']).to(self.device).double()
            lbls = sample_batch['label'].to(self.device)
            output = self.model(imgs)
            class_preds = torch.argmax(output, dim=1)
            running_loss += self.loss_criterion(output, lbls).item()
            acc += (class_preds == lbls).sum()
            del imgs
            del lbls
        acc /= self.val_len
        running_loss /= len(self.val_dataloader)
        print('\nAccuracy of the trained model: ', acc.item())
        print('Loss of the trained model: ', running_loss)
        self.save_model(acc, running_loss)

    def start_defence(self, defence_type = 'sign-indp'):
        if defence_type == 'yolo-mask':
            yolov5 = YOLOMask()
            #yolov5.read_test_results('yolov5/runs/train/yolo-patch-detection-sm-final/weights/best.pt', self.model)
            model_weight_path = ''
            test_df = yolov5.read_test_results('val', self.model)
            self.check_accuracy(test_df, self.val_len)
            #do something else
            #self.initialize_csv_for_bb()
            #self.prepare_train_val_txt()
        else:
            self.model.cpu()
            self.model.eval()
            sign_def = SignatureIndp(self.model)
            clean_acc = 0.0
            attacked_acc = 0.0
            defence_acc = 0.0
            for batch_i, batch in enumerate(self.val_dataloader):
                attack_predictions, defence_predictions = \
                sign_def.run(batch['image'], batch['label'])
                clean_class = torch.argmax(self.model(self.normalizer(batch['image'])), dim=1)
                clean_acc += (clean_acc == batch['label']).sum()
                attacked_acc += (attack_predictions == batch['label']).sum()
                defence_acc += (defence_predictions == batch['label']).sum()
            clean_acc /= len(self.val_dataloader)
            attacked_acc /= len(self.val_dataloader)
            defence_acc /= len(self.val_dataloader)
            print(f'Results with {self.model_name}')
            print('Clean image accuracy: ', clean_acc)
            print('Attacked image accuracy: ', attacked_acc)
            print('Defence image accuracy: ', defence_acc)

    def prepare_train_val_txt(self):
        for split in ['train', 'val']:
            patch_path = os.path.join(os.getcwd(),
            f'{self.patch_dir}/{split}/images')
            files = os.listdir(patch_path)
            print(f'No of files in {split}', len(files))
            lst_file = os.path.join(os.getcwd(), f'{self.patch_dir}/{split}.txt')
            with open(lst_file, "w") as fp:
                for line in files:
                    fpath = os.path.join(patch_path, line)
                    fp.write("".join(fpath) + "\n")

    def save_model(self, acc, loss):
        print("\nSaving model")
        model_path = os.path.join(os.getcwd(), f'models/retrained-classifiers/{self.model_name}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'acc': acc,
            'val_loss': loss
        }, model_path)

    def load_model(self):
        print("\nLoading pretrained model")
        model_path = os.path.join(os.getcwd(), f'models/retrained-classifiers/{self.model_name}.pt')
        model_chkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_chkpoint['model_state_dict'])
        self.model.eval()
    
    def initialize_csv_for_bb(self):
        df = pd.DataFrame([], columns=['filepath', 'xmin','ymin', 'xmax', 'ymax'])
        data_dir = os.path.join(os.getcwd(), 'patched-images')
        df.to_csv(os.path.join(data_dir, 'patch_bbs_train.csv'), index = False)
        df.to_csv(os.path.join(data_dir, 'patch_bbs_val.csv'), index = False)


if __name__ == "__main__":

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--rescale_size', default=224, type=int)
    parser.add_argument('--model_name', default='resnet18')
    parser.add_argument('--operation', default='train-classifier')
    parser.add_argument('--attack_shape', default='square')
    parser.add_argument('--defence_type', default='yolo-mask')
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--test_dir', default='yolov5/runs/detect/exp2/labels')
    parser.add_argument('--subset-size', default=-1, type=int)
    args = parser.parse_args()
    index = Index(args.model_name)
    index.start_program(args.rescale_size, args.crop_size,
    args.operation, args.defence_type, args.epochs,
    args.test_dir, args.subset_size)

