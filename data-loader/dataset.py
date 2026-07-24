import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):#runs once -> load the data
        self.X = torch.tensor([
            [1.0],
            [2.0],
            [3.0],
            [4.0]
        ])
        self.Y = torch.tensor([
            [2.0],
            [4.0],
            [6.0],
            [8.0]
        ])
    def __len__(self):#return dataset size
        return len(self.X)

    def __getitem__(self, index): #returns one sample
        return self.X[index], self.Y[index]

3
#creating dataset
dataset = MyDataset()

