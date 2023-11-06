from torchvision.io import write_png
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision.utils import draw_bounding_boxes
import pandas as pd

class PatchAttack:
    INPUT_SIZE = 224
    PATCH_SIZE = 50
    model = None
    device = "cpu"
    use_cuda = False
    mode = 'train'
    output_dir = ''
    current_dir = ''
    lbl_le_mapping = {}

    def __init__(self, model, lbl_le_mapping, output_dir, use_cuda = False):
        self.model = model
        self.output_dir = output_dir
        self.lbl_le_mapping = lbl_le_mapping
        if use_cuda:
          self.use_cuda = use_cuda
          self.device = "cuda"
          self.model = self.model.cuda()

    def __denormalize(self, img_batch):
      mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
      std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)
      return (img_batch * std) + mean

    def __apply_patch_box_on_img(self, batch_size, patch_batch, batch_img, filepaths):
        bounding_boxes = []
        for i in range(batch_size):
          randx = torch.randint(low=0, high=self.INPUT_SIZE - self.PATCH_SIZE, size=(1,)).item()
          randy = torch.randint(low=0, high=self.INPUT_SIZE - self.PATCH_SIZE, size=(1,)).item()
          xmax, ymax = randx + self.PATCH_SIZE, randy + self.PATCH_SIZE
          box_coord = torch.Tensor([[randx, randy, xmax, ymax]])
          batch_img[i, :, randy: ymax, randx: xmax] = (255 * self.__denormalize(patch_batch[i, :, :self.PATCH_SIZE, :self.PATCH_SIZE].data))
          write_png(((batch_img[0])).to(torch.uint8), 'sample_patch_img0.JPEG')
          batch_img[i] = draw_bounding_boxes(image=batch_img[i].to(torch.uint8),
          boxes=box_coord, colors="red")
          fn = f'{filepaths[i].replace(".JPEG", "")}_patchedimg.JPEG'
          df_row = [fn] + box_coord.tolist()[0]
          bounding_boxes.append(df_row)
        return batch_img, bounding_boxes

    def __apply_patch_on_batch(self, patch_batch, batch_size, minthresh, maxthresh):
      for i in range(batch_size):
        patch_batch[i].data = patch_batch[i].data.clamp(min=minthresh[i], max=maxthresh[i])
      return patch_batch

    def __patch_attack(self, image_batch_orig, mask_orig, target_class, max_iterations=100, target_thres=0.8, loss_fn=nn.CrossEntropyLoss()):
        target_class = target_class.to(device=self.device)
        batch_size = image_batch_orig.shape[0]

        patch_batch = torch.rand(image_batch_orig.shape)
        patch_batch.requires_grad_()

        patch_batch = nn.Parameter(patch_batch)
        optimizer = torch.optim.SGD([patch_batch], lr = 10, momentum=0.9)

        img_max = torch.amax(image_batch_orig, dim=(1,2,3))
        img_min = torch.amin(image_batch_orig, dim=(1,2,3))

        for i in range(max_iterations):
            mask = mask_orig.clone().detach()
            image = image_batch_orig.clone().detach()

            optimizer.zero_grad()

            patch_batch.data = self.__apply_patch_on_batch(
            patch_batch,
            batch_size,
            img_max, img_min
            )
            #write_png(patch_batch[0].to(torch.uint8), 'sample_patch_img.jpeg')
            patched_batch = mask *  patch_batch + (1-mask) * image
            if self.use_cuda:
              patched_batch = patched_batch.type(torch.cuda.FloatTensor)
            else:
              patched_batch = patched_batch.type(torch.FloatTensor)
            patched_batch = patched_batch.to(device=self.device)
            output = self.model(patched_batch)
            
            loss = loss_fn(output, target_class)
            loss.backward()
            optimizer.step()

            #patch_batch.data = patch_batch.data.clamp(min=img_min, max=img_max)
            patch_batch.data = self.__apply_patch_on_batch(
              patch_batch,
              batch_size,
              img_max, img_min
            )
            #write_png(patch_batch[0].to(torch.uint8), 'sample_patch_img.jpeg')
            
            del mask
            del image

            patch_batch.grad.zero_()

            pred = output.squeeze(0)
            pred = nn.Softmax(dim=0)(pred)
            cols = target_class.tolist()

            if pred[range(batch_size), cols].mean().item() > target_thres:
                break

        return  mask_orig * patch_batch + (1-mask_orig) * image_batch_orig
        #return patch_batch

    def __generate_mask(self, batch_size = 64, x=0, y=0, l=100, shape="square"):
        mask = torch.zeros([batch_size, 3, self.INPUT_SIZE, self.INPUT_SIZE])
        if shape == "square":
            mask[:, :, :self.PATCH_SIZE, :self.PATCH_SIZE] = 1
        # elif shape == "circle":
        '''
        else:
            x, y = x + l/2, y+l/2
            for i in range(mask.shape[1]):
                for j in range(mask.shape[2]):
                    if (y-i)**2 + (x-j)**2 < (l**2)/4:
                        mask[:, i, j] = 1
        '''
        return mask
    
    def add_patch_to_img(self, img_batch, lbls, filenames, mode = 'train'):
        #write_png((255 * self.__denormalize(img_batch[0])).to(torch.uint8), 'sample_input_img.JPEG')
        batch_size = img_batch.size()[0]
        self.current_dir = os.path.join(self.output_dir, mode)
        mask = self.__generate_mask(batch_size)
        patch_batch = self.__patch_attack(img_batch, mask, lbls)
        patch_batch_dnorm = 255 * self.__denormalize(patch_batch.data)
        img_batch_dnorm = 255 * self.__denormalize(img_batch)
        #write_png(patch_batch_dnorm[0].to(torch.uint8), 'sample_patch.JPEG')
        img_batch_res, bounding_boxes = self.__apply_patch_box_on_img(batch_size,
          patch_batch_dnorm, img_batch_dnorm, filenames)
        #write_png((img_batch_res[0]).to(torch.uint8), 'sample_patch_img.JPEG')

        # save csv file of bounding boxes
        bb_df = pd.read_csv(os.path.join(self.output_dir, f'patch_bbs_{mode}.csv'))
        bb_df = pd.concat([pd.DataFrame(bounding_boxes, columns=bb_df.columns),
        bb_df], ignore_index=True)
        bb_df.to_csv(os.path.join(self.output_dir, f'patch_bbs_{mode}.csv'),
          index = False)

        #test patch and save
        for i in range(batch_size):
          fn = filenames[i]
          tc = self.lbl_le_mapping[lbls[i].item()]
          dirpath = os.path.join(self.current_dir, tc)
          if not(os.path.exists(dirpath)):
            os.mkdir(dirpath)
          fn = os.path.join(self.current_dir, f'{filenames[i].replace(".JPEG","")}_patchedimg.JPEG')
          write_png(img_batch_res[i].to(torch.uint8), os.path.join(dirpath, fn))