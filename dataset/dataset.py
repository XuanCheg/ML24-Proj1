import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset_name, mode='train'):
        self.dataset_name = dataset_name
        self.data_root = f'data/{dataset_name}'
        self.data_path = os.path.join(self.data_root, mode + '.npz')
        data = np.load(self.data_path)

        label = list(data.files)
        self.data = []
        self.label = []
        for i, l in enumerate(label):
            self.data.append(data[l])
            self.label.extend([i] * len(data[l]))
        self.data = np.concatenate(self.data, axis=0)
        self.label = np.array(self.label)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].astype(np.int16)).float(), torch.tensor(self.label[idx]).long()


if __name__ == '__main__':
    dataset_name = 'ADNI'
    dataset = CustomDataset(dataset_name)
    print(len(dataset))
    print(dataset[0])
