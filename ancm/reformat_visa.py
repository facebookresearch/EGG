import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=4)
args = parser.parse_args()
n_distractors = args.n

def reshape_make_tensor(data, n_distractors, n_features):
    labels = []
    data_tensor = torch.tensor(data)
    data_reshaped = torch.empty((len(data), n_distractors + 1, n_features), dtype=torch.float32)
    for item in range(len(data)):
        # create a list of size of distractors
        distractor_idx = list(range(0, n_distractors+1))
        # randomly pick an index for the actual data and remove that from list and save in labels
        idx = np.random.randint(0,n_distractors+1)
        distractor_idx.remove(idx)
        labels.append(idx)
        # add feature data to tensor at correct idx
        data_reshaped[item][idx] = data_tensor[item]

        for distractor in distractor_idx:
            picked_features = item
            while picked_features == item:
                picked_features = np.random.randint(0, len(data))
            data_reshaped[item][distractor] = data_tensor[picked_features]

    return data_reshaped, labels

visa = pd.read_csv("visa.csv")

features = visa.iloc[:, 1:].values

# we do not really need thos I think
textlabels = visa.iloc[:, :1].values

n_features = features.shape[1]

# divide 60% for train, 20% test and valid
train_features, temp_features, train_textlabels, temp_labels = train_test_split(features, textlabels, test_size=0.4)
valid_features, test_features, valid_textlabels, test_textlabels = train_test_split(temp_features, temp_labels, test_size=0.5)

train_size = len(train_features)
valid_size = len(valid_features)
test_size = len(test_features)

train, train_labels = reshape_make_tensor(train_features, n_distractors, n_features)
valid, valid_labels = reshape_make_tensor(valid_features, n_distractors, n_features)
test, test_labels = reshape_make_tensor(test_features, n_distractors, n_features)

np.savez(f"visa-{n_distractors+1}", train=train, valid=valid, test=test,
         train_labels=train_labels, valid_labels=valid_labels, test_labels=test_labels,
         n_distractors=n_distractors)
_distractors = args.n
