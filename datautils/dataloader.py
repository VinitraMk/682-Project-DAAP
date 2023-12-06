import torch
from torch.utils.data import DataLoader, Subset

def get_dataloader(traindataset, valdataset, subset_len = 128, get_subset = False):
    if get_subset:
        random_els = torch.randint(0, len(traindataset), (subset_len,)).tolist()
        trainset = torch.utils.data.Subset(traindataset, random_els)
        random_els = torch.randint(0, len(valdataset), (subset_len,)).tolist()
        valset = torch.utils.data.Subset(valdataset, random_els)
    else:
        trainset = traindataset
        valset = valdataset
    
    val_len = len(valset)
    train_len = len(trainset)
    train_dataloader = DataLoader(trainset, batch_size=64,
                            shuffle=True, num_workers=2)
    val_dataloader = DataLoader(valset, batch_size=64,
                            shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader, train_len, val_len
