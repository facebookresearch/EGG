import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F


# GS classes
class SenderGS(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(SenderGS, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        return self.fc1(x).tanh()


class ReceiverGS(nn.Module):
    def __init__(self, n_features, linear_units):
        super(ReceiverGS, self).__init__()
        self.fc1 = nn.Linear(n_features, linear_units)

    def forward(self, x, _input, _aux_input=None):
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()


def loss_gs(_sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {"acc": acc}


# Reinforce
class SenderReinforce(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(SenderReinforce, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        return self.fc1(x).tanh()


class ReceiverReinforce(nn.Module):
    def __init__(self, n_features, linear_units):
        super(ReceiverReinforce, self).__init__()
        self.fc1 = nn.Linear(n_features, linear_units)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x, _input, _aux_input=None):
        embedded_input = self.fc1(_input).tanh()
        dots = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        dots = dots.squeeze()
        return self.logsoft(dots)


def loss_reinforce(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax() == _labels).detach().float()
    return -acc, {'acc': acc}
