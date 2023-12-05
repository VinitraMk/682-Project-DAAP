import torch

class SignatureIndp():
    init_seed_size = 50
    expansion_size = 20
    seed_reg_size = 0
    start_x = -1
    start_y = -1
    end_x = -1
    end_y = -1

    def __init__():
        pass
        

    def __initialize_seed_region(self):
        self.start_x, self.start_y = torch.randint(0, 113), torch.randint(0, 113)
        if self.seed_reg_size == 0:
            self.seed_reg_size = self.init_seed_size
        self.end_x = self.start_x + self.seed_reg_size
        self.end_y = self.start_y + self.seed_reg_size

    def __expand_img(self, bx, by, ex, ey):
        ex += self.expansion_size
        ey += self.expansion_size
        bx -= self.expansion_size
        by -= self.expansion_size
        is_full_image = False

        if (bx < 0):
            bx = 0
        if (by < 0):
            by = 0
        if (ex > 223):
            ex = 223
        if (ey > 223):
            ey = 223
        
        if bx == 0 and by == 0 and ex == 223 and ey == 223:
            is_full_image = True
        return bx, by, ex, ey, is_full_image



    def __start_test(self):
        is_full_img_region = False
        bx, by = self.start_x, self.start_y
        ex, ey = self.end_x, self.end_y 

        while not(is_full_img_region):
            img_whitewashed = 
            bx, by, ex, ey, is_full_img_region = self.__expand_img(bx, by, ex, ey)



    def start():
        print('run program')

if __name__ == "__main__":
    sign_indp = SignatureIndp()
    sign_indp.start()
