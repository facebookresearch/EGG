import argparse
import torch
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def reshape_make_list_of_tensors(data_concepts):

    data_concepts = np.array(data_concepts.iloc[:,1:].values, dtype='int')
    data_reshaped = []
    for concept_i in range(len(data_concepts)):
        data_reshaped.append(data_concepts[concept_i])

    return data_reshaped

def reformat():
    np.random.seed(42)
    visa = pd.read_csv("egg\\zoo\\adjusted_channel_game\\visa.csv")

    features = visa.iloc[:, 1:]
    textlabels = visa.iloc[:, :1]
    n_features = features.shape[1] - 1  # exclude the category column

    # divide 60% for train, 20% test and valid
    train_features, temp_features, train_textlabels, temp_labels = train_test_split(features, textlabels, test_size=0.40)
    valid_features, test_features, valid_textlabels, test_textlabels = train_test_split(temp_features, temp_labels, test_size=0.5)

    train_size = len(train_features)
    valid_size = len(valid_features)
    test_size = len(test_features)

    train = reshape_make_list_of_tensors(train_features)
    valid = reshape_make_list_of_tensors(valid_features)
    test = reshape_make_list_of_tensors(test_features)
   
    return train, valid, test



