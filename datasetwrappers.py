import torch.utils.data as data
import torch

class TimeArrowWrapper(data.Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        X, _ = self.dataset[idx]
        
        do_reverse = bool(random.getrandbits(1))
        
        if do_reverse:
            X = torch.flip(X, dims=(0, ))
        return X, int(do_reverse)
