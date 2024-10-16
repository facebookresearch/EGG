import torch
import numpy as np

class _OneHotIterator:
    def __init__(self, one_hot_list, n_batches_per_epoch, batch_size, seed=None):
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size
        self.one_hot_list = one_hot_list
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        # Sample batch_size one-hot vectors from the list
        indices = self.random_state.choice(len(self.one_hot_list), size=self.batch_size)
        batch_data = [self.one_hot_list[i] for i in indices]
        self.batches_generated += 1

        batch_data_np = np.array(batch_data)

        # Now convert it into a tensor
        batch_tensor = torch.tensor(batch_data_np).float()

        
        # Convert to tensor and return
        return batch_tensor, torch.zeros(1)

class OneHotLoader(torch.utils.data.DataLoader):
    def __init__(self, one_hot_list, batches_per_epoch, batch_size, seed=None):
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.one_hot_list = one_hot_list
        self.batch_size = batch_size

    def __iter__(self):
        if self.seed is None:
            seed = 42
        else:
            seed = self.seed

        return _OneHotIterator(
            one_hot_list=self.one_hot_list,
            n_batches_per_epoch=self.batches_per_epoch,
            batch_size=self.batch_size,
            seed=seed,
        )
