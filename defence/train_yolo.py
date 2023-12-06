from __future__ import division

from models import *
from utils.utils import *
from dataset import *
#from utils.parse_config import *

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


cuda = torch.cuda.is_available()

os.makedirs("checkpoints", exist_ok=True)

classes = [0, 1]

# Get data configuration
train_path = os.path.join(os.getcwd(), 'patched-images/train')
val_path = os.path.join(os.getcwd(), 'patched-images/valid')

# Get hyper parameters
learning_rate = 0.002
momentum = 0.9
decay = 0.0005
burn_in = 1000
checkpoint_interval = 25
checkpoint_dir = 'checkpoints'
epochs = 100

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
#model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=64, shuffle=False, num_workers=0
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))