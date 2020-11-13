# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


class DSpritesDataset(Dataset):
    def __init__(self, path_to_data, subsample=100, transform=None, image=False):
        self.imgs = np.load(path_to_data, encoding='bytes')['imgs'][::subsample]
        self.labels = np.load(path_to_data, encoding='bytes')['latents_classes'][::subsample]
        self.latent_values = np.load(path_to_data, encoding='bytes')['latents_values'][::subsample]

        self.transform = transform
        self.image = image

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.imgs[idx]*255
        latent_values = self.latent_values[idx] # latents_values: (737280 x 6, float64) Values of the latent factors.

        if self.image:
            image = image.reshape(image.shape + (1,))
            if self.transform:
                image = self.transform(image)

            # drop color - not a factor of variation
            latent_values = np.array(latent_values[1:])
            label = np.array(label[1:])

            # one hotify the categorical variable (shape)
            b = np.zeros(3)
            b[int(latent_values[0] - 1)] = 1

            latent_values = np.concatenate((b, latent_values[1:]), axis=0)

            return image.float(), torch.from_numpy(latent_values).unsqueeze(1).float(), torch.from_numpy(label).unsqueeze(1).float()

        else: # compo vs gen game
            return np.array(latent_values, dtype='float32'), label


def get_dsprites_dataloader(batch_size=32,
                            validation_split=.1,
                            random_seed=42,
                            shuffle=True,
                            path_to_data='dsprites.npz',
                            subsample=100,
                            image=False):

    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor(),
                                    subsample=subsample,
                                    image=image)

    dataset_size = len(dsprites_data)

    print("Dataset size: ", dataset_size)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                               sampler=train_sampler, drop_last=True)
    validation_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                                    sampler=valid_sampler, drop_last=True)
    return train_loader, validation_loader

