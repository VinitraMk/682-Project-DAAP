# starter code goes here
import utils
from cleanup import cleanup_images
from patch_attack import PatchAttack
from models.resnet18 import Resnet18
from transforms.transforms import Rescale, ToTensor, CenterCrop, Normalize

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

class Index:
    data_dir = ''
    output_dir = ''
    train_labels_path = ''
    lbl_le_mapping = {}
    model = None

    def __init__(self):
        self.data_dir = 'data/imagenette2'
        self.patch_dir = 'patched-images'
        self.output_dir = 'output'
        self.train_labels_path = self.data_dir + '/train_images.csv'
        self.val_labels_path = self.data_dir + '/val_images.csv'

    def start_program(self):
        # clean up gray scale images from the directory [temporary solution]
        print('program started')
        #cleanup_images(self.data_dir)
        #utils.make_csv(self.data_dir)
        self.build_datasetloader()
        self.initialize_model()
        #self.initialize_csv_for_bb()
        self.add_patches_imgs()
        #self.prepare_train_val_txt()

        
    def build_datasetloader(self):
        print('building datasets...')
        self.lbl_le_mapping = utils.make_csv(self.data_dir)

        data_transforms = tvtransforms.Compose([
            Rescale(256),
            ToTensor(),
            CenterCrop(224)])

        batch_size = 64

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
        if model_name == 'resnet18':
            self.model = Resnet18()
            self.model.train()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001, betas=(0.85, 0.888))


    def __insert_patch_on_batch(self, img_batch, lbls, filenames):
        with open(os.path.join(os.getcwd(), "data/class_id2name.json")) as f:
           target_to_classname = eval(f.read())

        with gzip.open(os.path.join(os.getcwd(), "data/imagenet_patch.gz"), "rb") as f:
             imagenet_patch = pickle.load(f)

        patches, targets, info = imagenet_patch
        patch = patches[0]
        self.model(img_batch, lbls)
        self.model.eval()

        apply_patch = PatchAttack(self.model, 'output', patch,
                                 (0.3, 0.3),
                                 80,
                                 (0.7, 1.5), info['patch_size'])

        normalizer = tvNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        patch_normalizer = Compose([apply_patch, normalizer])

        
        #preprocess = Compose([Resize(256), CenterCrop(224), ToTensor()])    # ensure images are 224x224
        #dataset = ImageFolder(os.path.join(os.getcwd(), "assets/data"),
                            #transform=preprocess,
                            #target_transform=utils.get_reduced_class_transforms)
        #data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
        #x, y = next(iter(data_loader))  # load a mini-batch
        x_clean = normalizer(img_batch)
        x_attacked = patch_normalizer(img_batch)
        #self.__display_patch(patches)
        self.model.double() 
        op_clean = self.model(x_clean.double())
        preds_clean = torch.argmax(op_clean, dim=1, keepdims=True).cpu().detach().numpy()
        op_attacked = self.model(x_attacked.double())
        preds_attacked =  torch.argmax(op_attacked, dim=1, keepdims=True).cpu().detach().numpy()
        pdx = pd.DataFrame({'Filenames': filenames, 'True Label': lbls, 'Clean Predictions': preds_clean[:, 0], 'Attacked Predictions': preds_attacked[:, 0]})
        pdx.to_csv(os.path.join(os.getcwd(), 'output/train_results.csv'))

    def add_patches_imgs(self):
        print('introduce patches to images..')
        output_dir = os.path.join(os.getcwd(), self.patch_dir)
        #print(torch.cuda.is_available())
        #patch_attack = PatchAttack(self.model,
        #self.lbl_le_mapping, output_dir, torch.cuda.is_available())
        self.model.double()
        self.model.train()
        for i in range(10):
            for i_batch, sample_batch in enumerate(self.train_dataloader):
                print(f'Adding patches to batch {i_batch}')
                self.optimizer.zero_grad()
                output = self.model(sample_batch['image'].double())
                loss = self.loss_criterion(output, sample_batch['label'])
                loss.backward()
                self.optimizer.step()

        self.model.eval()
        print('\nAdding patches to valid data\n')
        for i_batch, sample_batch in enumerate(self.val_dataloader):
            print(f'Adding patches to batch {i_batch}')
            self.__insert_patch_to_img(sample_batch['image'],
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
