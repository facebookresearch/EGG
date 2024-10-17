import os
import argparse
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, default=4)
    parser.add_argument("-s", type=int, default=5)
    parser.add_argument("-c", type=str, default=None)
    args = parser.parse_args()
    n_distractors = args.d
    n_samples = args.s
    return n_distractors, n_samples


def reshape_make_tensor(data_concepts, n_distractors, n_features, n_samples, category=None, data_distractors=None):
    if data_distractors is None:
        data_distractors = data_concepts

    labels = []
    categories = data_concepts.iloc[:,0]
    data_concepts = np.array(data_concepts.iloc[:,1:].values, dtype='int')
    data_reshaped = torch.empty((len(data_concepts)*n_samples, n_distractors + 1, n_features))
    for concept_i in range(len(data_concepts)):
        for sample_j in range(n_samples):
            target_pos = np.random.randint(0, n_distractors+1)
            distractor_pos = [i for i in range(n_distractors) if i != target_pos]
            data_reshaped[concept_i + sample_j, target_pos, :] = torch.tensor(data_concepts[concept_i])
               
            # randomly pick other distractors
            # remove the category column
            # the concept should be picked as its own distractor
            distractors = np.delete(data_distractors.iloc[:, 1:], [concept_i], axis=0)
            distractor_ids = np.random.choice(distractors.shape[0], n_distractors, replace=False)
            distractors = distractors[distractor_ids,:]
                
            for distractor_k, distractor_pos in enumerate(distractor_pos):
                if distractor_pos == target_pos:
                    continue
                data_reshaped[concept_i + sample_j, distractor_pos] = torch.tensor(distractors[distractor_k])

            labels.append(target_pos)

    return data_reshaped, labels


def get_filename(n_distractors, n_samples, category=None, extension=False):
    filename = f"data/input_data/visa-{n_distractors+1}-{n_samples}" 
    if category:
        filename += f'-{category}'
    if extension:
        filename += '.npz'
    return filename


def is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all()


def reformat(n_distractors, n_samples, category=None, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    ds = 'homonyms' if category is not None else 'no-homonyms'
    visa = pd.read_csv(f"data/visa-{ds}.csv")

    if category:
        visa = visa[visa.category == category.lower()]
   
    for column in visa.columns:
        if column != 'category' and is_unique(visa[column]):
            del visa[column]

    features = visa.iloc[:, 1:]
    textlabels = visa.iloc[:, :1]
    n_features = features.shape[1] - 1  # exclude the category column

    # divide 70% for train, 15% test and valid
    train_features, temp_features, train_textlabels, temp_labels = train_test_split(features, textlabels, test_size=0.3)
    valid_features, test_features, valid_textlabels, test_textlabels = train_test_split(temp_features, temp_labels, test_size=0.5)

    train_size = len(train_features)
    valid_size = len(valid_features)
    test_size = len(test_features)

    train, train_labels = reshape_make_tensor(train_features, n_distractors, n_features, n_samples, category)
    valid, valid_labels = reshape_make_tensor(valid_features, n_distractors, n_features, n_samples, category)
    test, test_labels = reshape_make_tensor(test_features, n_distractors, n_features, n_samples, category)
  
    print('Exporting VISA...\n\ncategory:', category if category else 'all', '\n\nnumber of samples:')
    print('  train:', len(train_labels))
    print('  val:', len(valid_labels))
    print('  test:', len(test_labels))

    print('\n\nNumber of features:', n_features)

    filename = get_filename(n_distractors, n_samples, category)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, train=train, valid=valid, test=test,
             train_labels=train_labels, valid_labels=valid_labels, test_labels=test_labels,
             n_distractors=n_distractors)
    print('dataset saved to ' + f"{filename}.npz")


def main():
    n_distractors, n_samples, category = parse_args()
    reformat(n_distractors, n_samples, category)


if __name__ == '__main__':
    main()
