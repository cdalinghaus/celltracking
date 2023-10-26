import numpy as np
import matplotlib.pyplot as plt
import torchvision
import random
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
import math
import torch

class PatchTransform:
    
    def __init__(self, patch_size, frame_size, num_channels):
        self.patch_size = patch_size
        self.frame_size = frame_size
        self.num_frames = None
        self.num_channels = num_channels
        
    def __call__(self, X):
        batch_size, num_frames, num_channels, x_width, y_width = X.shape
        self.num_frames = num_frames
        num_patches = x_width // self.patch_size
        patch_size = self.patch_size
        
        examples = []
        for example in X:
            patches = []
            for frame in example:
                for y in range(num_patches):
                    for x in range(num_patches):
                        patch = frame[:, x*patch_size:x*patch_size+patch_size, y*patch_size:y*patch_size+patch_size]
                        patches.append(patch)
            patches = torch.stack(patches, axis=0)
            examples.append(patches)
        examples = torch.stack(examples, axis=0)
        return examples
    
    def invert(self, X):
        batch_size, num_patches_times_num_frames, num_channels, patch_size, patch_size = X.shape
        patches_per_frame = int((self.frame_size / self.patch_size) * (self.frame_size / self.patch_size))
        patches_per_axis = int(self.frame_size / self.patch_size)
        
        examples = []
        for example in X:
            frames = torch.zeros(self.num_frames, self.num_channels, self.frame_size, self.frame_size)
            for patch_index, patch in enumerate(example):
                frame_index = patch_index // patches_per_frame
                frame_offset = patch_index % patches_per_frame
                
                frame_pointer_x = frame_offset // patches_per_axis
                frame_pointer_y = frame_offset % patches_per_axis
                #print(frame_pointer_x, frame_pointer_y)
                
                frames[frame_index, :, frame_pointer_y*self.patch_size:frame_pointer_y*self.patch_size+self.patch_size, frame_pointer_x*self.patch_size:frame_pointer_x*self.patch_size+self.patch_size] = patch

            examples.append(frames)
        return torch.stack(examples, axis=0)