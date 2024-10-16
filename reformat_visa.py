import argparse
import torch
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_distractors", type=int, default=4)
    parser.add_argument("-n_samples", type=int, default=5)
    args = parser.parse_args()
    n_distractors = args.n_distractors
    n_samples = args.n_samples
    return n_distractors, n_samples


def reshape_make_tensor(data_concepts, n_distractors, n_features, n_samples, data_distractors=None):
    if data_distractors is None:  # if no df for distractors is provided, use df for concepts
        data_distractors = data_concepts

    labels = []
    categories = data_concepts.iloc[:,0]
    data_concepts = np.array(data_concepts.iloc[:,1:].values, dtype='int')
    data_reshaped = torch.empty((len(data_concepts)*n_samples, n_distractors + 1, n_features), dtype=torch.float32)
    for concept_i in range(len(data_concepts)):
        category = categories.iloc[concept_i]  # data_concepts.iloc[concept_i,0]
        for sample_j in range(n_samples):
            target_pos = np.random.randint(0, n_distractors+1)
            distractor_pos = [i for i in range(n_distractors) if i != target_pos]
            data_reshaped[concept_i + sample_j, target_pos, :] = torch.tensor(data_concepts[concept_i])

            
            #if sample_j <= math.floor(sample_j / n_samples) \
                    #and len(data_distractors[data_distractors.category == category]) >= math.ceil(1.5 * n_samples): 
                # if there are sufficiently many concepts in the category,
                # pick random distractors from the same category
     
                #distractors = data_distractors[data_distractors.category == category] 
                #distractors = distractors.sample(n=n_distractors).iloc[:, 1:]
                #distractors = np.array(distractors, dtype='int')
            
            #else:  #  sample_j <= math.floor(sample_j / n_samples):
               
                # randomly pick distractors
            distractors = data_distractors.iloc[:, 1:]  # remove the category column
            distractors = distractors.sample(n=n_distractors)
            distractors = np.array(distractors, dtype='int')

            for distractor_k, distractor_pos in enumerate(distractor_pos):
                if distractor_pos == target_pos:
                    continue
                data_reshaped[concept_i + sample_j, distractor_pos] = torch.tensor(distractors[distractor_k])

            labels.append(target_pos)

    return data_reshaped, labels


def reformat(n_distractors, n_samples):
    np.random.seed(42)
    visa = pd.read_csv("ancm/visa.csv")

    # features = visa.iloc[:, 1:].values
    # textlabels = visa.iloc[:, :1].values
    # categories = visa.iloc[:, 1:2].values

    features = visa.iloc[:, 1:]
    textlabels = visa.iloc[:, :1]
    n_features = features.shape[1] - 1  # exclude the category column

    # divide 60% for train, 20% test and valid
    train_features, temp_features, train_textlabels, temp_labels = train_test_split(features, textlabels, test_size=0.27)
    valid_features, test_features, valid_textlabels, test_textlabels = train_test_split(temp_features, temp_labels, test_size=0.5)

    train_size = len(train_features)
    valid_size = len(valid_features)
    test_size = len(test_features)

    train, train_labels = reshape_make_tensor(train_features, n_distractors, n_features, n_samples, features)
    valid, valid_labels = reshape_make_tensor(valid_features, n_distractors, n_features, n_samples, features)
    test, test_labels = reshape_make_tensor(test_features, n_distractors, n_features, n_samples, features)
   
    print('train:', len(train_labels))
    print('val:', len(valid_labels))
    print('test:', len(test_labels))

    np.savez(f"visa-{n_samples}", train=train, valid=valid, test=test,
             train_labels=train_labels, valid_labels=valid_labels, test_labels=test_labels,
             n_distractors=n_distractors)


def main():
    n_distractors, n_samples = parse_args()
    reformat(n_distractors, n_samples)


if __name__ == '__main__':
    main()
