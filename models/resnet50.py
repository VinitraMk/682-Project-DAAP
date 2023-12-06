import torch.nn as nn
import torch.optim as optim

from models.ots_models import get_model

class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _ = get_model('resnet50')
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
    def forward(self, x):
        return self.model(x)

            
