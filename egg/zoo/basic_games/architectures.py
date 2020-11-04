import torch.nn as nn
import torch


class RecoReceiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(RecoReceiver, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input):
        return self.output(x)

class DiscriReceiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(DiscriReceiver, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _input):
        embedded_input = self.fc1(_input).tanh()
        dots = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return dots.squeeze()

class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        return self.fc1(x)
        # here, it might make sense to add a non-linearity, such as tanh
