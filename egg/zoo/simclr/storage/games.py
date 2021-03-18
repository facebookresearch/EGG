"""
from egg.core.reinforce_wrappers import (
    RnnSenderReinforce,
    RnnReceiverDeterministic,
    SenderReceiverRnnReinforce,
)

from egg.core.continous_communication import (
    SenderReceiverContinuousCommunication
)


def get_continuous_opts(parser):
    group = parser.add_argument_group("continuous")
    group.add_argument(
        "--sender_output_size",
        type=int,
        default=128,
        help="Sender output size and message dimension in continuous communication"
    )



def get_rf_opts(parser):
    group = parser.add_argument_group("reinforce")
    # sender opts
    group.add_argument(
        "--recurrent_cell",
        type=str,
        default="rnn",
        choices=["rnn", "lstm", "gru"],
        help="Type of the cell used for Sender and Receiver {rnn, gru, lstm} (default: rnn)"
    )
    group.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=0.1,
        help="Entropy regularisation coeff for Sender (default: 0.1)"
    )
    group.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Sender (default: 10)"
    )
    group.add_argument(
        "--sender_rnn_num_layers",
        type=int,
        default=1,
        help="Number hidden layers of sender. Only in reinforce (default: 1)"
    )
    # receiver opts
    group.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Receiver (default: 10)"
    )
    group.add_argument(
        "--receiver_rnn_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)"
    )
    group.add_argument(
        "--receiver_rnn_num_layers",
        type=int,
        default=1,
        help="Number hidden layers of receiver. Only in reinforce (default: 1)"
    )





class ContinuousSender(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128
    ):
        super(ContinuousSender, self).__init__()

        self.agent = agent
        self.fwd = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        out = self.agent(x)
        return self.fwd(out)

def build_sender_receiver_rf(
    sender,
    receiver,
    vocab_size,
    max_len,
    sender_embed_dim,
    receiver_embed_dim,
    sender_hidden,
    receiver_hidden,
    sender_rnn_num_layers,
    receiver_rnn_num_layers,
    recurrent_cell,
):
    sender = RnnSenderReinforce(
        agent=sender,
        vocab_size=vocab_size,
        embed_dim=sender_embed_dim,
        hidden_size=sender_hidden,
        max_len=max_len,
        num_layers=sender_rnn_num_layers,
        cell=recurrent_cell
    )
    receiver = RnnReceiverDeterministic(
        receiver,
        vocab_size=vocab_size,
        embed_dim=receiver_embed_dim,
        hidden_size=receiver_hidden,
        cell=recurrent_cell,
        num_layers=receiver_rnn_num_layers
    )
    return sender, receiver

    if opts.communication_channel == "continuous":
        sender = ContinuousSender(
            agent=sender,
            input_dim=effective_projection_dim,
            hidden_dim=effective_projection_dim,
            output_dim=opts.sender_output_size
        )
        game = SenderReceiverContinuousCommunication(
            sender,
            receiver,
            loss,
            train_logging_strategy,
            test_logging_strategy
        )
    elif opts.communication_channel == "rf":
        sender, receiver = build_sender_receiver_rf(
            sender,
            receiver,
            vocab_size=opts.vocab_size,
            max_len=opts.max_len,
            sender_embed_dim=opts.sender_embedding,
            receiver_embed_dim=opts.receiver_embedding,
            sender_rnn_hidden=effective_projection_dim,
            receiver_rnn_hidden=opts.receiver_rnn_hidden,
            sender_rnn_num_layers=opts.sender_rnn_num_layers,
            receiver_rnn_num_layers=opts.receiver_rnn_num_layers,
            cell=opts.recurrent_cell,
        )
        game = SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            train_logging_strategy=train_logging_strategy,
            test_logging_strategy=test_logging_strategy
        )

"""
