import torch
import torch
import torchvision.transforms as transforms

class TuboletTransform:
    def __init__(self, tubolet_shape):
        # tubolet_shape should be (num_frames, num_x, num_y)
        # batch size and channel size will be inferred automatically
        # Input shape: (batch, frame, channel, width, height)
        assert len(tubolet_shape) == 3
        self.tubolet_shape = tubolet_shape
        
    def __call__(self, X):
        #print(X.shape)
        batch_size = X.shape[0]
        channel_size = X.shape[2]
        X = X.moveaxis(2, 4)
        #print(X.shape)
        tubolet_shape = (batch_size, ) + self.tubolet_shape + (channel_size, )
        #print(tubolet_shape)
        tubolet_number_of_elements_per_axis = tuple([X.shape[i]//x for i, x in enumerate(tubolet_shape)])
        tnoepa = tubolet_number_of_elements_per_axis

        res = torch.Tensor(np.lib.stride_tricks.sliding_window_view(X, window_shape=tubolet_shape))
        #print(res.shape)
        res = res[0, ::X.shape[1]//tnoepa[1], ::X.shape[2]//tnoepa[2], ::X.shape[3]//tnoepa[3]]
        #print(res.shape)
        res = res.moveaxis(3, 0)
        #print(res.shape)
        res = res.reshape(batch_size, -1, self.tubolet_shape[0], self.tubolet_shape[1], self.tubolet_shape[2], channel_size)
        
        
        return res
