import torch.nn as nn
import torch

# In EGG, a script designer must provide the core functionality of the Sender and Receiver agents. These are then
# embedded in wrappers that are used to train them to play Gumbel-Softmax- or Reinforce-optimized games. The core
# Sender must take the input and produce a hidden representation that is then used by the wrapper to initialize
# the RNN or other architecture that will generate the message. The core Receiver must expect a hidden representation
# generated by the message-processing wrapper, plus possibly other game-specific input, and it must generate the
# game-specific output.

# The RecoReceiver class implements the core Receiver agent for the recogntion game. This is simply a linear layer
# that takes as input the vector generated by the message-decoding RNN in the wrapper (x in the forward method) and
# produces an output of n_features dimensionality, to be interpreted as a one-hot representation of the reconstructed
# attribute-value vector
class RecoReceiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(RecoReceiver, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input):
        return self.output(x)

# The DiscriReceiver class implements the core Receiver agent for the reconstruction game. In this case, besides the
# vector generated by the message-decoding RNN in the wrapper (x) in the forward method), the module also gets game-specific
# Receiver input, that is, the matrix containing all input attribute-value vectors. The module maps these vectors to the
# same dimensionality as the RNN output vector, and computes a dot product between the latter and each of the (transformed) input vectors.
# The output dot prodoct list is interpreted as the non-normalized probability distribution over possible positions of the target.
class DiscriReceiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(DiscriReceiver, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _input):
        # the rationale for the non-linearity here is that the RNN output will also be the outcome of a non-linearity
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
