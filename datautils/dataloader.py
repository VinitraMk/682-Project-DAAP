import torch
from torch.utils.data import DataLoader, Subset

def get_dataloader(traindataset, valdataset, batch_size = 64,
    subset_len = 128, get_subset = False, num_workers = 0):
    if get_subset:
        random_els = torch.randint(0, len(traindataset), (subset_len,)).tolist()
        trainset = torch.utils.data.Subset(traindataset, random_els)
        val_len = int(0.4 * subset_len)
        random_els = torch.randint(0, len(valdataset), (val_len,)).tolist()
        valset = torch.utils.data.Subset(valdataset, random_els)
    else:
        trainset = traindataset
        valset = valdataset
    
    val_len = len(valset)
    train_len = len(trainset)
    train_dataloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(valset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    return train_dataloader, val_dataloader, train_len, val_len
