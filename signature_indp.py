import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch.nn as nn
from models.ots_models import get_model

class SignatureIndp():
    contractn_size = 20
    start_x = -1
    start_y = -1
    end_x = -1
    end_y = -1
    min_width = 100
    min_height = 100

    def __init__(self, model, target_H = 224, target_W = 224):
        #self.img = read_image('./test_img.JPEG')
        #_, self.H, self.W = self.img.size()
        self.H, self.W = target_H, target_W
        self.model = model
        

    def __initialize_seed_region(self):
        self.start_x , self.start_y = 0, 0
        self.end_x, self.end_y = self.W - 1, self.H - 1

    def __downsize_img(self, bx, by, ex, ey):
        nex = ex - self.contractn_size
        ney = ey - self.contractn_size
        nbx = bx - self.contractn_size
        nby = by - self.contractn_size
        is_min_image = False

        width = nex - nbx
        height = ney - nby
        if width < self.min_width:
            nbx = bx
            nex = nbx + self.min_width
        if height < self.min_height:
            nby = by
            ney = nby + self.min_height

        width = nex - nbx
        height = ney - nby

        if width == self.min_width or height == self.min_height:
            is_min_image = True
        return nbx, nby, nex, ney, is_min_image
    
    def __whitewash_img(self, bx, by, ex, ey, img):
        whitewash_mask = torch.clone(img)
        whitewash_mask[:, :, :, :bx] = 255
        whitewash_mask[:, :, :by, :] = 255
        whitewash_mask[:, :, :, ex:] = 255
        whitewash_mask[:, :, ey:, :] = 255
        return whitewash_mask


    def __start_test(self, X, y):
        attack_predictions = []
        defence_predictions = []
        for i in range(X.size()[0]):
            is_min_img = False
            bx, by = self.start_x, self.start_y
            ex, ey = self.end_x, self.end_y 
            img = X[i]
            print('sign indp', img.size())
            attacked_preds = self.model(img.unsqueeze(0))
            full_img_class = torch.argmax(attacked_preds, dim=1)
            print('full img class pred', full_img_class.size())
            attack_predictions.append(full_img_class)
            defence_class = -1
            while not(is_min_img):
                bx, by, ex, ey, is_full_img_region = self.__downsize_img(bx, by, ex, ey)
                img_whitewashed = self.__whitewash_img(bx, by, ex, ey, img).reshape((self.H, self.W, 3))
                attacked_preds = self.model(img_whitewashed)
                attacked_class = torch.argmax(attacked_preds, dim=1)
                if full_img_class != attacked_class:
                    defence_class = attacked_class
                    break
                #plt.clf() 
                #print(img_whitewashed.numpy().shape)
                #plt.imshow(img_whitewashed.numpy(), interpolation='nearest')
                #plt.axis(False)
                #plt.show()
            if defence_class != -1:
                defence_predictions.append(defence_class)
            else:
                defence_predictions.append(full_img_class)
        return attack_predictions, defence_predictions
            
    def run(self, X, y):
        print('\nRun signature independent adversarial patch detector program')
        self.__initialize_seed_region()
        self.__start_test(X, y)


if __name__ == "__main__":
    sign_indp = SignatureIndp()
    sign_indp.run()
