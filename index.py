# starter code goes here
from utils import utils
from cleanup import cleanup_images
from patch_attack import PatchAttack
from models.resnet18 import Resnet18
from transforms.transforms import Rescale, ToTensor, CenterCrop, Normalize
from defence.signature_indp import SignatureIndp

import os
import torch
from dataset import ImageNetDataset
from torchvision import transforms as tvtransforms
from torch.utils.data import DataLoader
import pandas as pd
import gzip
import pickle
from torchvision.transforms import Compose, Resize, CenterCrop as tvCenterDrop, ToTensor as tvToTensor, \
    Normalize as tvNormalize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse
from torchvision.utils import save_image
import random
import numpy as np

class Index:
    data_dir = ''
    output_dir = ''
    train_labels_path = ''
    lbl_le_mapping = {}
    model = None
    normalizer = tvNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    model_name = 'resnet18'

    def __init__(self):
        self.data_dir = 'data/imagenette2'
        self.patch_dir = 'attacked-images'
        self.output_dir = 'output'
        self.train_labels_path = self.data_dir + '/train_images.csv'
        self.val_labels_path = self.data_dir + '/val_images.csv'
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def start_program(self, defence_type = 'sign-indp'):
        # clean up gray scale images from the directory [temporary solution]
        print('program started')
        #cleanup_images(self.data_dir)
        #utils.make_csv(self.data_dir)
        self.build_datasetloader()
        self.initialize_model(self.model_name)
        #self.retrain_classifier()
        self.load_model()
        self.add_patches_to_imgs()
        #self.save_model()
        #self.start_defence(defence_type) 

        
    def build_datasetloader(self):
        print('building datasets...')
        self.lbl_le_mapping = utils.make_csv(self.data_dir)

        data_transforms = tvtransforms.Compose([
            Rescale(256),
            ToTensor(),
            CenterCrop(224)])

        batch_size = 128

        imagenette_train_dataset = ImageNetDataset(labels_path = self.train_labels_path,
                                                data_dir = os.path.join(self.data_dir, 'train'),
                                                transform=data_transforms,
                                                lbl_mapping=self.lbl_le_mapping)
        imagenette_val_dataset =  ImageNetDataset(labels_path = self.val_labels_path,
                                                data_dir = os.path.join(self.data_dir, 'val'),
                                                transform=data_transforms,
                                                lbl_mapping=self.lbl_le_mapping)
        print('\nLength of train dataset: ',len(imagenette_train_dataset))
        print('Length of val dataset: ',len(imagenette_val_dataset), '\n')
        self.train_dataloader = DataLoader(imagenette_train_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
        self.val_dataloader = DataLoader(imagenette_val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    def initialize_model(self, model_name = 'resnet18'):
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        

        self.model_name = model_name
        if model_name == 'resnet18':
            self.model = Resnet18()
            self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01, momentum=0.9)
        self.model.to(self.device)

    def __save_attacked_images(self, imgs, filenames, mode = 'train'):
        for i in range(len(imgs)):
            img = imgs[i].cpu()#.transpose(1, 2, 0)
            output_dir = os.path.join(os.getcwd(), f'attacked-images/{mode}')
            save_image(img, os.path.join(output_dir, f"{filenames[i]}"))

    def __insert_patch_on_batch(self, img_batch, lbls, filenames, mode = 'train'):
        filepath = os.path.join(os.getcwd(), f'attacked-images/{mode}/{mode}_results.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            df = pd.DataFrame({'Filenames': [], 'True Label': [], 'Clean Predictions': [], 'Attacked Predictions': []})
            df.to_csv(filepath)
        with open(os.path.join(os.getcwd(), "data/class_id2name.json")) as f:
           target_to_classname = eval(f.read())

        with gzip.open(os.path.join(os.getcwd(), "data/imagenet_patch.gz"), "rb") as f:
             imagenet_patch = pickle.load(f)

        patch_idx = torch.randint(0, len(patches), (1,))[0].item()
        patches, targets, info = imagenet_patch
        patch = patches[patch_idx]

        apply_patch = PatchAttack(self.model, 'output', patch,
                                 (0.3, 0.3),
                                 80,
                                 (0.7, 1.5), info['patch_size'])

        normalizer = tvNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        patch_normalizer = Compose([apply_patch, normalizer])

        img_batch.to(self.device)
        x_clean = normalizer(img_batch).to(self.device)
        x_attacked = patch_normalizer(img_batch).to(self.device)
        img_batch.cpu().detach()
        self.__save_attacked_images(x_attacked, filenames, mode)
        self.model.double()
        op_clean = self.model(x_clean.double())
        preds_clean = torch.argmax(op_clean, dim=1, keepdims=True).cpu().detach().numpy()
        op_attacked = self.model(x_attacked.double())
        preds_attacked =  torch.argmax(op_attacked, dim=1, keepdims=True).cpu().detach().numpy()
        x_clean.cpu().detach()
        x_attacked.cpu().detach()
        pdx = pd.DataFrame({'Filenames': filenames, 'True Label': lbls, 'Clean Predictions': preds_clean[:, 0], 'Attacked Predictions': preds_attacked[:, 0]})
        pdx = pd.concat([df, pdx], axis=0, ignore_index = True)
        pdx.to_csv(os.path.join(os.getcwd(), filepath))

    def add_patches_to_imgs(self):
        print('\nAttack images')
        self.model.eval()
        for i_batch, sample_batch in enumerate(self.train_dataloader):
            print(f'\tAdding patches to train batch {i_batch}')
            self.__insert_patch_on_batch(sample_batch['image'],
            sample_batch['label'],
            sample_batch['filename'], 'train')

        for i_batch, sample_batch in enumerate(self.val_dataloader):
            print(f'\tAdding patches to val batch {i_batch}')
            self.__insert_patch_on_batch(sample_batch['image'],
            sample_batch['label'],
            sample_batch['filename'], 'val')

    def retrain_classifier(self):
        print('Retraining classifier on 10 classes')
        output_dir = os.path.join(os.getcwd(), self.patch_dir)
        self.model = self.model.double()
        self.model.train()
        for i in range(10):
            print(f"Running epoch {i}")
            for i_batch, sample_batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                imgs = sample_batch['image'].to(self.device).double()
                lbls = sample_batch['label'].to(self.device)
                output = self.model(imgs)
                loss = self.loss_criterion(output, lbls)
                loss.backward()
                self.optimizer.step()
                lbls.cpu().detach()
                imgs.cpu().detach()

        self.save_model()

    def start_defence(self, defence_type = 'sign-indp'):
        print(defence_type)
        if defence_type == 'yolo-mask':
            #do something else
            #self.initialize_csv_for_bb()
            #self.prepare_train_val_txt()
            pass
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

    def save_model(self):
        print("\nSaving model")
        model_path = os.path.join(os.getcwd(), f'models/retrained-classifiers/{self.model_name}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, model_path)

    def load_model(self):
        print("\nLoading pretrained model")
        model_path = os.path.join(os.getcwd(), f'models/retrained-classifiers/{self.model_name}.pt')
        model_chkpoint = torch.load(model_path)
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
    parser.add_argument('--attack_shape', default='square')
    parser.add_argument('--defence_type', default='sign-indp')
    args = parser.parse_args()
    index = Index()
    index.start_program(args.defence_type)
