import matplotlib.pyplot as plt
import torch

def plotframes(X):
    batch_size, frames, color_channels, x, y = X.shape
    
    fig, ax = plt.subplots(1, frames, figsize=(20, 3))
    
    for idx in range(frames):
        ax.flat[idx].imshow(X[0, idx].T.cpu().detach().numpy())
    plt.show()