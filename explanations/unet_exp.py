import torch
import os
import tqdm
from matplotlib import pyplot as plt
from torchvision.io import read_image, write_png

from defenses.unet import ResnetUnet


def unnormalize(img):
    m = torch.Tensor([0.485, 0.456, 0.406])
    s = torch.Tensor([0.229, 0.224, 0.225])
    m = m.unsqueeze(1).unsqueeze(1)
    s = s.unsqueeze(1).unsqueeze(1)
    return img * s + m

def unet_feature_invert(model_save_dir, save_dir):
    use_cuda = True
    os.makedirs(save_dir, exist_ok=True)

    unet = ResnetUnet()
    unet.load_state_dict(
        torch.load(os.path.join(model_save_dir, "model.ckpt"))
    )
    unet.eval()
    if use_cuda:
        unet = unet.cuda()

    inverted_img = torch.rand((1, 3, 224, 224))
    labels = torch.ones((1, 1, 224, 224))
    zero = torch.Tensor([0])
    if use_cuda:
        inverted_img = inverted_img.cuda()
        labels = labels.cuda()
        zero = zero.cuda()
    inverted_img.requires_grad_()
    inverted_img = torch.nn.Parameter(inverted_img)
    inverted_img.requires_grad_()
    optimizer = torch.optim.SGD([inverted_img], lr = 10, momentum=0.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[2000, 5000, 8000], gamma=0.1
    )


    num_iters = 10000
    pbar = tqdm.tqdm(total = num_iters)
    iter2loss = []

    img_max = torch.Tensor([2.2489, 2.4286, 2.6400])
    img_min = torch.Tensor([-2.1179, -2.0357, -1.8044])
    img_max = img_max[None, :, None, None].cuda()
    img_min = img_min[None, :, None, None].cuda()

    for it in range(num_iters):
        loss_accum = []
        optimizer.zero_grad()

        
        predictions = unet(inverted_img)
        loss_bce = torch.nn.BCELoss()(
            predictions, labels
        ) 
        
        loss_reg = torch.nn.MSELoss()(inverted_img, zero)
        loss = loss_bce + 0.1*loss_reg
        loss.backward()
        optimizer.step()

        pbar.update(1)
        pbar.set_description("Loss: {:.4f}".format(loss.item()))
        iter2loss.append(loss.item())

        lr_scheduler.step()
        with torch.no_grad():
            inverted_img[:] = inverted_img.clamp(img_min, img_max)


        if it %100 == 0:
            
            plt.plot(iter2loss)
            plt.savefig(os.path.join(save_dir, "feature_invert_loss_unet.png"), dpi=300)
            plt.close()

            img_to_save = inverted_img.clone().detach().cpu()
            print(loss_bce.item(), loss_reg.item())
            print(img_to_save.max(), img_to_save.min())
            print("====")
            img_to_save = 255*unnormalize(img_to_save).squeeze(0)
            write_png(img_to_save.to(torch.uint8), os.path.join(save_dir, "inverted_img.png"))



