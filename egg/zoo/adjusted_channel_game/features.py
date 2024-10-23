import torch
import numpy as np

class _OneHotIteratorFromList:
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

class OneHotLoaderFromList(torch.utils.data.DataLoader):
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

        return _OneHotIteratorFromList(
            one_hot_list=self.one_hot_list,
            n_batches_per_epoch=self.batches_per_epoch,
            batch_size=self.batch_size,
            seed=seed,
        )
class _OneHotIterator:
    """
    >>> it_1 = _OneHotIterator(n_features=128, n_batches_per_epoch=2, batch_size=64, probs=np.ones(128)/128, seed=1)
    >>> it_2 = _OneHotIterator(n_features=128, n_batches_per_epoch=2, batch_size=64, probs=np.ones(128)/128, seed=1)
    >>> list(it_1)[0][0].allclose(list(it_2)[0][0])
    True
    >>> it = _OneHotIterator(n_features=8, n_batches_per_epoch=1, batch_size=4, probs=np.ones(8)/8)
    >>> data = list(it)
    >>> len(data)
    1
    >>> batch = data[0]
    >>> x, y = batch
    >>> x.size()
    torch.Size([4, 8])
    >>> x.sum(dim=1)
    tensor([1., 1., 1., 1.])
    >>> probs = np.zeros(128)
    >>> probs[0] = probs[1] = 0.5
    >>> it = _OneHotIterator(n_features=128, n_batches_per_epoch=1, batch_size=256, probs=probs, seed=1)
    >>> batch = list(it)[0][0]
    >>> batch[:, 0:2].sum().item()
    256.0
    >>> batch[:, 2:].sum().item()
    0.0
    """

    def __init__(self, n_features, n_batches_per_epoch, batch_size, probs, seed=None):
        self.n_batches_per_epoch = n_batches_per_epoch
        self.n_features = n_features
        self.batch_size = batch_size

        self.probs = probs
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        batch_data = self.random_state.multinomial(1, self.probs, size=self.batch_size)
        self.batches_generated += 1
        return torch.from_numpy(batch_data).float(), torch.zeros(1)


class OneHotLoader(torch.utils.data.DataLoader):
    """
    >>> probs = np.ones(8) / 8
    >>> data_loader = OneHotLoader(n_features=8, batches_per_epoch=3, batch_size=2, probs=probs, seed=1)
    >>> epoch_1 = []
    >>> for batch in data_loader:
    ...     epoch_1.append(batch)
    >>> [b[0].size() for b in epoch_1]
    [torch.Size([2, 8]), torch.Size([2, 8]), torch.Size([2, 8])]
    >>> data_loader_other = OneHotLoader(n_features=8, batches_per_epoch=3, batch_size=2, probs=probs)
    >>> all_equal = True
    >>> for a, b in zip(data_loader, data_loader_other):
    ...     all_equal = all_equal and (a[0] == b[0]).all()
    >>> all_equal.item()
    0
    """

    def __init__(self, n_features, batches_per_epoch, batch_size, probs, seed=None):
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.n_features = n_features
        self.batch_size = batch_size
        self.probs = probs

    def __iter__(self):
        if self.seed is None:
            seed = 42
        else:
            seed = self.seed

        return _OneHotIterator(
            n_features=self.n_features,
            n_batches_per_epoch=self.batches_per_epoch,
            batch_size=self.batch_size,
            probs=self.probs,
            seed=seed,
        )

class UniformLoader(torch.utils.data.DataLoader):
    def __init__(self, n_features):
        self.batch = torch.eye(n_features), torch.zeros(1)

    def __iter__(self):
        return iter([self.batch])

