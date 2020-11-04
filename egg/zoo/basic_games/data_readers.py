import torch
from torch.utils.data import Dataset
import numpy as np

class AttValRecoDataset(Dataset):
    def __init__(self, path,n_attributes,n_values):
        frame = np.loadtxt(path, dtype='S10')
        self.frame = []
        for row in frame:
            if (n_attributes==1):
                row = row.split()
            config = list(map(int, row))
            z=torch.zeros((n_attributes,n_values))
            for i in range(n_attributes):
                z[i,config[i]]=1
            label = torch.tensor(list(map(int, row)))
            self.frame.append((z.view(-1),label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

class AttValDiscriDataset(Dataset):
    def __init__(self, path,n_values):
        frame = open(path,'r')
        self.frame = []
        for row in frame:
            raw_info = row.split('.')
            index_vectors = list([list(map(int,x.split())) for x in raw_info[:-1]])
            target_index = int(raw_info[-1])
            target_one_hot = []
            for index in index_vectors[target_index]:
                current=np.zeros(n_values)
                current[index]=1
                target_one_hot=np.concatenate((target_one_hot,current))
            target_one_hot_tensor = torch.FloatTensor(target_one_hot)
            one_hot = []
            for index_vector in index_vectors:
                for index in index_vector:
                    current=np.zeros(n_values)
                    current[index]=1
                    one_hot=np.concatenate((one_hot,current))
            one_hot_sequence = torch.FloatTensor(one_hot).view(len(index_vectors),-1)
            label= torch.tensor(target_index)
            self.frame.append((target_one_hot_tensor,label,one_hot_sequence))
        frame.close()

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]
