import numpy as np
import torch
class ProximitySampler:
    """
    Samples batches of data based on proximity in a feature space.

    :param cos_sim_matrix: Path to the cosine similarity matrix file. numpy format.
    :param batch_size: The size of each batch.
    :param idxs: Optional indices to filter the cosine similarity matrix.
    """
    def __init__(self, cos_sim_matrix, batch_size, idxs=None):
        self.cos_sim_matrix = torch.tensor(np.load(cos_sim_matrix))
        if idxs is not None:
            self.cos_sim_matrix = self.cos_sim_matrix[idxs][:, idxs]
        self.batch_size = batch_size

    def __iter__(self):
        idxs = torch.tensor([])
        for i in range(len(self.cos_sim_matrix) // self.batch_size):
            ref_idx = torch.randint(0, len(self.cos_sim_matrix), (1,)).item()
            idxs = torch.cat(
                (
                    idxs,
                    torch.argsort(self.cos_sim_matrix[ref_idx], 0, descending=True)[1 : self.batch_size + 1]
                )
            )
        return (i for i in idxs.int())
    
    def __len__(self):
        return (len(self.cos_sim_matrix) // self.batch_size) * self.batch_size