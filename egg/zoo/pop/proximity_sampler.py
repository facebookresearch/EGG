import numpy as np
import torch
class ProximitySampler:
    # samples the batch_size closest images to the current image using the cosine similarity matrix
    def __init__(self, cos_sim_matrix, batch_size):
        self.cos_sim_matrix = np.load(cos_sim_matrix)
        self.batch_size = batch_size

    def __iter__(self):
        idxs = torch.tensor([])
        for i in range(len(self.cos_sim_matrix)):
            idxs = torch.cat(
                (
                    idxs,
                    torch.tensor(
                        [torch.argsort(self.cos_sim_matrix[i], descending=True)[1 : self.batch_size + 1]]
                    ),
                )
            )
        return iter(idxs)