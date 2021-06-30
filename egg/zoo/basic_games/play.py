# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents
from egg.zoo.basic_games.architectures import DiscriReceiver, RecoReceiver, Sender
from egg.zoo.basic_games.data_readers import AttValDiscriDataset, AttValRecoDataset


# the following section specifies parameters that are specific to our games: we will also inherit the
# standard EGG parameters from https://github.com/facebookresearch/EGG/blob/master/egg/core/util.py
def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments controlling the game type
    parser.add_argument(
        "--game_type",
        type=str,
        default="reco",
        help="Selects whether to play a reco(nstruction) or discri(mination) game (default: reco)",
    )
    # arguments concerning the input data and how they are processed
    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default=None, help="Path to the validation data"
    )
    # (the following is only used in the reco game)
    parser.add_argument(
        "--n_attributes",
        type=int,
        default=None,
        help="Number of attributes in Sender input (must match data set, and it is only used in reco game)",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        default=None,
        help="Number of values for each attribute (must match data set)",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=0,
        help="Batch size when processing validation data, whereas training data batch_size is controlled by batch_size (default: same as training data batch size)",
    )
    # arguments concerning the training method
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    # the following if statement controls aspects specific to the two game tasks: loss, input data and architecture of the Receiver
    # (the Sender is identical in both cases, mapping a single input attribute-value vector to a variable-length message)
    if opts.game_type == "discri":
        # the game object we will encounter below takes as one of its mandatory arguments a loss: a loss in EGG is expected to take as arguments the sender input,
        # the message, the Receiver input, the Receiver output and the labels (although some of these elements might not actually be used by a particular loss);
        # together with the actual loss computation, the loss function can return a dictionary with other auxiliary statistics: in this case, accuracy
        def loss(
            _sender_input,
            _message,
            _receiver_input,
            receiver_output,
            labels,
            _aux_input,
        ):
            # in the discriminative case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
            # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
            acc = (receiver_output.argmax(dim=1) == labels).detach().float()
            # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
            loss = F.cross_entropy(receiver_output, labels, reduction="none")
            return loss, {"acc": acc}

        # the input data are read into DataLodaer objects, which are pytorch constructs implementing standard data processing functionalities, such as shuffling
        # and batching
        # within our games, we implement dataset classes, such as AttValDiscriDataset, to read the input text files and convert the information they contain
        # into the form required by DataLoader
        # look at the definition of the AttValDiscrDataset (the class to read discrimination game data) in data_readers.py for further details
        # note that, for the training dataset, we first instantiate the AttValDiscriDataset object and then feed it to DataLoader, whereas for the
        # validation data (confusingly called "test" data due to code heritage inertia) we directly declare the AttValDiscriDataset when instantiating
        # DataLoader: the reason for this difference is that we need the train_ds object to retrieve the number of features of the input vectors
        train_ds = AttValDiscriDataset(path=opts.train_data, n_values=opts.n_values)
        train_loader = DataLoader(
            train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=1
        )
        test_loader = DataLoader(
            AttValDiscriDataset(path=opts.validation_data, n_values=opts.n_values),
            batch_size=opts.validation_batch_size,
            shuffle=False,
            num_workers=1,
        )
        # note that the number of features retrieved here concerns inputs after they are converted to 1-hot vectors
        n_features = train_ds.get_n_features()
        # we define here the core of the Receiver for the discriminative game, see the architectures.py file for details:
        # note that this will be embedded in a wrapper below to define the full agent
        receiver = DiscriReceiver(n_features=n_features, n_hidden=opts.receiver_hidden)

    else:  # reco game

        def loss(
            sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
        ):
            # in the case of the recognition game, for each attribute we compute a different cross-entropy score
            # based on comparing the probability distribution produced by the Receiver over the values of each attribute
            # with the corresponding ground truth, and then averaging across attributes
            # accuracy is instead computed by considering as a hit only cases where, for each attribute, the Receiver
            # assigned the largest probability to the correct value
            # most of this function consists of the usual pytorch madness needed to reshape tensors in order to perform these computations
            n_attributes = opts.n_attributes
            n_values = opts.n_values
            batch_size = sender_input.size(0)
            receiver_output = receiver_output.view(batch_size * n_attributes, n_values)
            receiver_guesses = receiver_output.argmax(dim=1)
            correct_samples = (
                (receiver_guesses == labels.view(-1))
                .view(batch_size, n_attributes)
                .detach()
            )
            acc = (torch.sum(correct_samples, dim=-1) == n_attributes).float()
            labels = labels.view(batch_size * n_attributes)
            loss = F.cross_entropy(receiver_output, labels, reduction="none")
            loss = loss.view(batch_size, -1).mean(dim=1)
            return loss, {"acc": acc}

        # again, see data_readers.py in this directory for the AttValRecoDataset data reading class
        train_loader = DataLoader(
            AttValRecoDataset(
                path=opts.train_data,
                n_attributes=opts.n_attributes,
                n_values=opts.n_values,
            ),
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=1,
        )
        test_loader = DataLoader(
            AttValRecoDataset(
                path=opts.validation_data,
                n_attributes=opts.n_attributes,
                n_values=opts.n_values,
            ),
            batch_size=opts.validation_batch_size,
            shuffle=False,
            num_workers=1,
        )
        # the number of features for the Receiver (input) and the Sender (output) is given by n_attributes*n_values because
        # they are fed/produce 1-hot representations of the input vectors
        n_features = opts.n_attributes * opts.n_values
        # we define here the core of the receiver for the discriminative game, see the architectures.py file for details
        # this will be embedded in a wrapper below to define the full architecture
        receiver = RecoReceiver(n_features=n_features, n_hidden=opts.receiver_hidden)

    # we are now outside the block that defined game-type-specific aspects of the games: note that the core Sender architecture
    # (see architectures.py for details) is shared by the two games (it maps an input vector to a hidden layer that will be use to initialize
    # the message-producing RNN): this will also be embedded in a wrapper below to define the full architecture
    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features)

    # now, we instantiate the full sender and receiver architectures, and connect them and the loss into a game object
    # the implementation differs slightly depending on whether communication is optimized via Gumbel-Softmax ('gs') or Reinforce ('rf', default)
    if opts.mode.lower() == "gs":
        # in the following lines, we embed the Sender and Receiver architectures into standard EGG wrappers that are appropriate for Gumbel-Softmax optimization
        # the Sender wrapper takes the hidden layer produced by the core agent architecture we defined above when processing input, and uses it to initialize
        # the RNN that generates the message
        sender = core.RnnSenderGS(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
        )
        # the Receiver wrapper takes the symbol produced by the Sender at each step (more precisely, in Gumbel-Softmax mode, a function of the overall probability
        # of non-eos symbols upt to the step is used), maps it to a hidden layer through a RNN, and feeds this hidden layer to the
        # core Receiver architecture we defined above (possibly with other Receiver input, as determined by the core architecture) to generate the output
        receiver = core.RnnReceiverGS(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
        game = core.SenderReceiverRnnGS(sender, receiver, loss)
        # callback functions can be passed to the trainer object (see below) to operate at certain steps of training and validation
        # for example, the TemperatureUpdater (defined in callbacks.py in the core directory) will update the Gumbel-Softmax temperature hyperparameter
        # after each epoch
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else:  # NB: any other string than gs will lead to rf training!
        # here, the interesting thing to note is that we use the same core architectures we defined above, but now we embed them in wrappers that are suited to
        # Reinforce-based optmization
        sender = core.RnnSenderReinforce(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0,
        )
        callbacks = []

    # we are almost ready to train: we define here an optimizer calling standard pytorch functionality
    optimizer = core.build_optimizer(game.parameters())
    # in the following statement, we finally instantiate the trainer object with all the components we defined (the game, the optimizer, the data
    # and the callbacks)
    if opts.print_validation_events == True:
        # we add a callback that will print loss and accuracy after each training and validation pass (see ConsoleLogger in callbacks.py in core directory)
        # if requested by the user, we will also print a detailed log of the validation pass after full training: look at PrintValidationEvents in
        # language_analysis.py (core directory)
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                core.PrintValidationEvents(n_epochs=opts.n_epochs),
            ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [core.ConsoleLogger(print_train_loss=True, as_json=True)],
        )

    # and finally we train!
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
