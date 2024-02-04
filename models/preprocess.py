import torch
import numpy as np

#two types of one image
class AAMixDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __getitem__(self, i):
        x, y = self.dataset[i]
        im_tuple = (self.preprocess(x), np.array(x).astype(np.uint8))
        return im_tuple, y

    def __len__(self):
        return len(self.dataset)