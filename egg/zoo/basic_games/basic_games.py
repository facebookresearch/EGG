import argparse

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch

import egg.core as core
from egg.core import Callback, Interaction

import numpy as np

def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments controlling the game type
    parser.add_argument('--game_type', type=str, default='reco',
                        help="Selects whether to play a reco(nstruction) or discri(mination) game (default: reco)")
    # arguments concerning the input data and how they are processed
    parser.add_argument('--train_data', type=str, default=None,
                        help='Path to the train data')
    parser.add_argument('--validation_data', type=str, default=None,
                        help='Path to the validation data')
    # (the following is only used in the reco game)
    parser.add_argument('--n_attributes', type=int, default=None,
                        help='Number of attributes in Sender input (must match data set, and it is only used in reco game)')
    parser.add_argument('--n_values', type=int, default=None,
                        help='Number of values for each attribute (must match data set)')
    parser.add_argument('--validation_batch_size', type=int, default=1000,
                        help='Batch size when processing validation data, whereas training data batch_size is controlled by batch_size (default: 1000)')
    # arguments concerning the training method
    parser.add_argument('--mode', type=str, default='rf',
                        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)")
    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)')
    # arguments concerning the agent architectures
    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)')
    # arguments controlling the script output
    parser.add_argument('--print_validation_events', default=False, action='store_true',
                        help='If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced bytthe Sender, and the output probabilities produced by the Receiver (default: do not print)')
    # NB: the script also inherits the default EGG arguments (https://github.com/mbaroni/EGG/blob/basic_games/egg/core/util.py)
    args = core.init(parser,params)
    return args


class AttValRecoDataset(Dataset):
    def __init__(self, path,n_attributes,n_values):
        frame = np.loadtxt(path, dtype='S10')
        self.frame = []
        for row in frame:
            if (n_attributes==1):
                row = row.split()
            config = list(map(int, row))
            z=torch.zeros((n_attributes,n_values))
            for i in range(n_attributes):
                z[i,config[i]]=1
            label = torch.tensor(list(map(int, row)))
            self.frame.append((z.view(-1),label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

class AttValDiscriDataset(Dataset):
    def __init__(self, path,n_values):
        frame = open(path,'r')
        self.frame = []
        for row in frame:
            raw_info = row.split('.')
            index_vectors = list([list(map(int,x.split())) for x in raw_info[:-1]])
            target_index = int(raw_info[-1])
            target_one_hot = []
            for index in index_vectors[target_index]:
                current=np.zeros(n_values)
                current[index]=1
                target_one_hot=np.concatenate((target_one_hot,current))
            target_one_hot_tensor = torch.FloatTensor(target_one_hot)
            one_hot = []
            for index_vector in index_vectors:
                for index in index_vector:
                    current=np.zeros(n_values)
                    current[index]=1
                    one_hot=np.concatenate((one_hot,current))
            one_hot_sequence = torch.FloatTensor(one_hot).view(len(index_vectors),-1)
            label= torch.tensor(target_index)
            self.frame.append((target_one_hot_tensor,label,one_hot_sequence))
        frame.close()

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

    
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

class PrintInMsgOut(Callback):
    def __init__(self, n_epochs):
        super().__init__()
        self.n_epochs=n_epochs
    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if (epoch==self.n_epochs):
            print("TRAIN_INS")
            print([m.tolist() for m in logs.sender_input], sep='\n')
            print("TRAIN_LABELS")
            print([m.tolist() for m in logs.labels], sep='\n')
            print("TRAIN_MSGS")
            print([m.tolist() for m in logs.message], sep='\n')
            print("TRAIN_OUTS")
            print([m.tolist() for m in logs.receiver_output], sep='\n')

    def on_test_end(self, _loss, logs: Interaction, epoch: int):
        if (epoch==self.n_epochs):
            print("VALID_INS")
            print([m.tolist() for m in logs.sender_input], sep='\n')
            print("VALID_LABELS")
            print([m.tolist() for m in logs.labels], sep='\n')
            print("VALID_MSGS")
            print([m.tolist() for m in logs.message], sep='\n')
            print("VALID_OUTS")
            print([m.tolist() for m in logs.receiver_output], sep='\n')
 #           print("reshaped receiver output")
 #           logs.receiver_output.view(100 * 3, 5)
# Roberto's original
# class PrintMsg(Callback):
#     def __init__(self, n_msgs: int = 10):
#         super().__init__()
#         assert n_msgs > 0
#         self.n_msgs = n_msgs
        
#     def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
#         assert self.n_msgs < logs.message.shape[0]
#         print([m.tolist() for m in logs.message[:self.n_msgs]], sep='\n')
  

def main(params):
    opts = get_params(params)
    print(opts, flush=True)

    def loss(sender_input, _message, _receiver_input, receiver_output, labels):
        n_attributes=opts.n_attributes
        n_values=opts.n_values
        batch_size = sender_input.size(0)
        receiver_output = receiver_output.view(batch_size*n_attributes, n_values)
        receiver_guesses = receiver_output.argmax(dim=1)
        correct_samples = (receiver_guesses == labels.view(-1)).view(batch_size,n_attributes).detach()
        acc = (torch.sum(correct_samples, dim=-1) == n_attributes).float()
        labels = labels.view(batch_size*n_attributes)
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        loss = loss.view(batch_size,-1).mean(dim=1)
        return loss, {'acc': acc}
    
    train_loader = DataLoader(AttValRecoDataset(path=opts.train_data,
                                         n_attributes=opts.n_attributes,
                                         n_values=opts.n_values),
                              batch_size=opts.batch_size,
                              shuffle=True, num_workers=1)
    test_loader = DataLoader(AttValRecoDataset(path=opts.validation_data,
                                        n_attributes=opts.n_attributes,
                                        n_values=opts.n_values),
                             batch_size=opts.validation_batch_size,
                             shuffle=False, num_workers=1)

    n_features=opts.n_attributes*opts.n_values
    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features)
    receiver = RecoReceiver(n_features=n_features, n_hidden=opts.receiver_hidden)

    if opts.mode.lower() == 'gs':
        sender = core.RnnSenderGS(sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_embedding, hidden_size=opts.sender_hidden, cell=opts.sender_cell, max_len=opts.max_len, temperature=opts.temperature)
        receiver = core.RnnReceiverGS(receiver, vocab_size=opts.vocab_size, embed_dim=opts.receiver_embedding, hidden_size=opts.receiver_hidden, cell=opts.receiver_cell)
        game = core.SenderReceiverRnnGS(sender, receiver, loss)
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else: # NB: any other string than gs will lead to rf training!
        sender = core.RnnSenderReinforce(sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_embedding, hidden_size=opts.sender_hidden, cell=opts.sender_cell, max_len=opts.max_len)
        receiver = core.RnnReceiverDeterministic(receiver, vocab_size=opts.vocab_size, embed_dim=opts.receiver_embedding, hidden_size=opts.receiver_hidden, cell=opts.receiver_cell)
        game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,receiver_entropy_coeff=0)
        callbacks = []
        
    optimizer = core.build_optimizer(game.parameters())
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,validation_data=test_loader,callbacks=callbacks+[core.ConsoleLogger(print_train_loss=True, as_json=True),PrintInMsgOut(n_epochs=opts.n_epochs)])
    #trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,validation_data=test_loader,callbacks=callbacks+[core.ConsoleLogger(print_train_loss=True, as_json=True)])

    trainer.train(n_epochs=opts.n_epochs)

    core.close()
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

