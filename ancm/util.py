import json
import torch
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from archs import ErasureChannel

from typing import Optional

from egg.core.util import move_to
from egg.zoo.objects_game.util import mutual_info, entropy


def compute_alignment(dataloader, sender, receiver, device, bs):
    all_features = dataloader.dataset.list_of_tuples
    targets = dataloader.dataset.target_idxs
    obj_features = np.unique(all_features[:,targets[0],:], axis=0)
    obj_features = torch.tensor(obj_features, dtype=torch.float).to(device)

    n_batches = math.ceil(obj_features.size()[0]/bs)
    sender_embeddings, receiver_embeddings = None, None

    for batch in [obj_features[bs*y:bs*(y+1),:] for y in range(n_batches)]:
        with torch.no_grad():
            b_sender_embeddings = sender.fc1(batch).tanh().numpy()
            b_receiver_embeddings = receiver.fc1(batch).tanh().numpy()
            if sender_embeddings is None:
                sender_embeddings = b_sender_embeddings
                receiver_embeddings = b_receiver_embeddings
            else:
                sender_embeddings = np.concatenate((sender_embeddings, b_sender_embeddings))
                receiver_embeddings = np.concatenate((receiver_embeddings, b_receiver_embeddings))

    sender_sims = cosine_similarity(sender_embeddings)
    receiver_sims = cosine_similarity(receiver_embeddings)
    r = pearsonr(sender_sims.ravel(), receiver_sims.ravel())
    return r.statistic * 100


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def dump_sender_receiver(
    game: torch.nn.Module,
    dataset: "torch.utils.data.DataLoader",
    gs: bool,
    apply_noise: bool,
    variable_length: bool,
    device: Optional[torch.device] = None,
):
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param gs: whether Gumbel-Softmax relaxation was used during training
    :param variable_length: whether variable-length communication is used
    :param device: device (e.g. 'cuda') to be used
    :return:
    """
    train_state = game.training  # persist so we restore it back
    game.eval()

    device = device if device is not None else common_opts.device

    sender_inputs, messages, receiver_inputs, receiver_outputs = [], [], [], []
    labels = []

    with torch.no_grad():
        for batch in dataset:
            # by agreement, each batch is (sender_input, labels) plus optional (receiver_input)
            sender_input = move_to(batch[0], device)
            receiver_input = None if len(batch) == 2 else move_to(batch[2], device)

            message = game.sender(sender_input)

            # Under GS, the only output is a message; under Reinforce, two additional tensors are returned.
            # We don't need them.
            if not gs:
                message = message[0]

            # Add noise to the message
            message = game.channel(message, apply_noise=apply_noise)

            output = game.receiver(message, receiver_input)
            if not gs:
                output = output[0]

            if batch[1] is not None:
                labels.extend(batch[1])

            if isinstance(sender_input, list) or isinstance(sender_input, tuple):
                sender_inputs.extend(zip(*sender_input))
            else:
                sender_inputs.extend(sender_input)

            if receiver_input is not None:
                receiver_inputs.extend(receiver_input)

            if gs:
                message = message.argmax(
                    dim=-1
                )  # actual symbols instead of one-hot encoded

            if not variable_length:
                messages.extend(message)
                receiver_outputs.extend(output)
            else:
                # A trickier part is to handle EOS in the messages. It also might happen that not every message has EOS.
                # We cut messages at EOS if it is present or return the entire message otherwise. Note, EOS id is always
                # set to 0.

                for i in range(message.size(0)):
                    eos_positions = (message[i, :] == 0).nonzero()
                    message_end = (
                        eos_positions[0].item() if eos_positions.size(0) > 0 else -1
                    )
                    assert message_end == -1 or message[i, message_end] == 0
                    if message_end < 0:
                        messages.append(message[i, :])
                    else:
                        messages.append(message[i, : message_end + 1])

                    if gs:
                        receiver_outputs.append(output[i, message_end, ...])
                    else:
                        receiver_outputs.append(output[i, ...])

    game.train(mode=train_state)

    return sender_inputs, messages, receiver_inputs, receiver_outputs, labels


def compute_mi_input_msgs(sender_inputs, messages):
    num_dimensions = len(sender_inputs[0])
    each_dim = [[] for _ in range(num_dimensions)]
    result = []
    for i, _ in enumerate(each_dim):
        for vector in sender_inputs:
            each_dim[i].append(vector[i])  # only works for 1-D sender inputs

    for i, dim_list in enumerate(each_dim):
        result.append(round(mutual_info(messages, dim_list), 4))

    return {
        'entropy_msg': entropy(messages),
        'entropy_inp': entropy(sender_inputs),
        'mi': mutual_info(messages, sender_inputs),
        'entropy_inp_dim': [entropy(elem) for elem in each_dim],
        'mi_dim': result,
    }
