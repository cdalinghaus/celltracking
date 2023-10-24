from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os


class GameDS(Dataset):
    
    def __init__(self):
        self.files = sorted(os.listdir("../frames"))
        self.sequence_length = 10

    def __len__(self):
        return len(self.files) - self.sequence_length
    
    def __getitem__(self, idx):
        files = self.files[idx:idx+self.sequence_length]
        
        images = [Image.open("../frames/" + f) for f in files]
        tensors = [torch.tensor(np.array(im)) for im in images]
        stacked = torch.stack(tensors)
        result = stacked
        result = result.moveaxis(3, 1)
        return result, torch.Tensor([0])
