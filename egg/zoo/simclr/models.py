# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_resnet(name, pretrained=False):
    """Loads ResNet encoder from torchvision along with features number"""
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")

    model = resnets[name]
    n_features = model.fc.in_features
    model.fc = Identity()
    return model, n_features


class LogisticRegression(nn.Module):
    """
    Simple Logistic Regression module.
    To be used for linear probes.
    """

    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)
