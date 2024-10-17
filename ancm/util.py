import json
import torch
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


def compute_alignment(dataloader, sender, receiver, device, bs):
    all_features = dataloader.dataset.list_of_tuples
    targets = dataloader.dataset.target_idxs
    obj_features = np.unique(all_features[:,targets[0],:], axis=0)
    obj_features = torch.tensor(obj_features, dtype=torch.float).to(device)

    n_batches = math.ceil(obj_features.size()[0]/bs)
    sender_embeddings, receiver_embeddings = None, None

    for batch in [obj_features[bs*y:bs*(y+1),:] for y in range(n_batches)]:
        with torch.no_grad():
            b_sender_embeddings = sender.fc1(batch).tanh().numpy()
            b_receiver_embeddings = receiver.fc1(batch).tanh().numpy()
            if sender_embeddings is None:
                sender_embeddings = b_sender_embeddings
                receiver_embeddings = b_receiver_embeddings
            else:
                sender_embeddings = np.concatenate((sender_embeddings, b_sender_embeddings))
                receiver_embeddings = np.concatenate((receiver_embeddings, b_receiver_embeddings))

    sender_sims = cosine_similarity(sender_embeddings)
    receiver_sims = cosine_similarity(receiver_embeddings)
    r = pearsonr(sender_sims.ravel(), receiver_sims.ravel())
    return r.statistic * 100


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
