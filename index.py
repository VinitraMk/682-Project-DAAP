# starter code goes here
import utils
from cleanup import cleanup_images
from patch_attack import PatchAttack
from models.ots_models import get_model

import os
import torch
from dataset import ImageNetDataset
from transforms import Rescale, ToTensor, CenterCrop, Normalize
from torchvision import transforms 
from torch.utils.data import DataLoader
import pandas as pd

class Index:
    data_dir = ''
    train_labels_path = ''
    lbl_le_mapping = {}
    model = None

    def __init__(self):
        self.data_dir = 'data/imagenette2'
        self.patch_dir = 'patched-images'
        self.train_labels_path = self.data_dir + '/train_images.csv'
        self.val_labels_path = self.data_dir + '/val_images.csv'

    def start_program(self):
        # clean up gray scale images from the directory [temporary solution]
        print('program started')
        #cleanup_images(self.data_dir)
        '''
        #utils.make_csv(self.data_dir)
        self.build_datasetloader()
        self.initialize_model()
        self.initialize_csv_for_bb()
        #self.add_patches_imgs()
        '''
        #self.prepare_train_val_txt()

        
    def build_datasetloader(self):
        print('building datasets...')
        self.lbl_le_mapping = utils.make_csv(self.data_dir)

        data_transforms = transforms.Compose([
            Rescale(256),
            ToTensor(),
            CenterCrop(224),
            Normalize()])

        batch_size = 64

        imagenette_train_dataset = ImageNetDataset(labels_path = self.train_labels_path,
                                                data_dir = os.path.join(self.data_dir, 'train'),
                                                transform=data_transforms,
                                                lbl_mapping=self.lbl_le_mapping)
        imagenette_val_dataset =  ImageNetDataset(labels_path = self.val_labels_path,
                                                data_dir = os.path.join(self.data_dir, 'val'),
                                                lbl_mapping=self.lbl_le_mapping,
                                                transform=data_transforms)
        print('\nLength of train dataset: ',len(imagenette_train_dataset))
        print('Length of val dataset: ',len(imagenette_val_dataset), '\n')
        self.train_dataloader = DataLoader(imagenette_train_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
        self.val_dataloader = DataLoader(imagenette_val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    def initialize_model(self):
        self.model, _ = get_model()
        self.model.eval()

    def add_patches_imgs(self):
        print('introduce patches to images..')
        output_dir = os.path.join(os.getcwd(), self.patch_dir)
        #print(torch.cuda.is_available())
        patch_attack = PatchAttack(self.model,
        self.lbl_le_mapping, output_dir, torch.cuda.is_available())
        print('\nAdding patches to training data\n')
        for i_batch, sample_batch in enumerate(self.train_dataloader):
            if i_batch >= 48:
                print(f'Adding patches to batch {i_batch}')
                patch_attack.add_patch_to_img(sample_batch['image'],
                sample_batch['label'],
                sample_batch['filename'], 'train')
        print('\nAdding patches to valid data\n')
        for i_batch, sample_batch in enumerate(self.val_dataloader):
            print(f'Adding patches to batch {i_batch}')
            patch_attack.add_patch_to_img(sample_batch['image'],
            sample_batch['label'],
            sample_batch['filename'], 'val')

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
    
    def initialize_csv_for_bb(self):
        df = pd.DataFrame([], columns=['filepath', 'xmin','ymin', 'xmax', 'ymax'])
        data_dir = os.path.join(os.getcwd(), 'patched-images')
        df.to_csv(os.path.join(data_dir, 'patch_bbs_train.csv'), index = False)
        df.to_csv(os.path.join(data_dir, 'patch_bbs_val.csv'), index = False)


if __name__ == "__main__":

    index = Index()
    index.start_program()
