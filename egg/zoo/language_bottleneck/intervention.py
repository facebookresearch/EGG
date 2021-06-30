# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy as np
import torch

import egg.core as core
from egg.core.batch import Batch


def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())

    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)


def entropy(messages):
    from collections import defaultdict

    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except ValueError:
        t = tuple(t.view(-1).tolist())
    return t


def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = entropy(xys)

    return e_x + e_y - e_xy


def _find_lengths(messages):
    """
    >>> messages = torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = _find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    positions = []
    for i in range(messages.size(0)):
        zero_pos = torch.nonzero(messages[i, :] == 0)
        if zero_pos.size(0) > 0:
            position = zero_pos[0].item() + 1
        else:
            position = messages.size(1)
        positions.append(position)
    return torch.Tensor(positions).long().to(messages.device)


class CallbackEvaluator(core.Callback):
    def __init__(
        self, dataset, device, is_gs, loss, var_length, input_intervention=False
    ):
        self.dataset = dataset
        self.is_gs = is_gs
        self.device = device
        self.loss = loss
        self.var_length = var_length
        self.input_intervention = input_intervention
        self.epoch = 0

    def intervention_message(self, game):
        mean_acc = 0.0
        scaler = 0.0

        bob_label_mi = 0.0

        corresponding_labels = []
        original_messages = []
        bob_inputs = []
        alice_inputs = []

        for batch in self.dataset:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)
            sender_input, labels, receiver_input, aux_input = batch

            original_message = game.sender(sender_input, aux_input)
            # if Reinforce, agents return tuples
            if not self.is_gs:
                original_message = original_message[0]

            if receiver_input is not None:
                bob_inputs.extend(receiver_input)
            alice_inputs.extend(sender_input)

            permutation = torch.randperm(original_message.size(0)).to(
                original_message.device
            )
            message = torch.index_select(original_message, 0, permutation)
            output = game.receiver(message, receiver_input, aux_input)

            if not self.is_gs:
                output = output[0]

            if not self.var_length:
                _, rest = self.loss(None, None, None, output, labels, aux_input)
                mean_acc += rest["acc"].mean().item()
                scaler += 1

                original_messages.extend(original_message)
            elif not self.is_gs:
                lengths = core.find_lengths(message)

                for i in range(lengths.size(0)):
                    curr_len = lengths[i]
                    original_messages.append(message[i, :curr_len])

                _, rest = self.loss(None, None, None, output, labels)
                mean_acc += rest["acc"].mean().item()
                scaler += 1
            else:
                message = message.argmax(dim=-1)
                lengths = core.find_lengths(message)

                for i in range(lengths.size(0)):
                    curr_len = lengths[i]
                    original_messages.append(message[i, :curr_len])

                    _, rest = self.loss(
                        None,
                        None,
                        None,
                        output[i : i + 1, curr_len - 1],
                        labels[i : i + 1],
                    )
                    mean_acc += rest["acc"].item()
                    scaler += 1

            corresponding_labels.extend(labels)

        label_entropy = entropy(corresponding_labels)

        message_info = mutual_info(original_messages, corresponding_labels)
        if bob_inputs:
            bob_label_mi = mutual_info(bob_inputs, corresponding_labels)
        alice_label_mi = mutual_info(alice_inputs, corresponding_labels)

        mean_acc /= scaler

        s = dict(
            mean_acc=mean_acc,
            label_entropy=label_entropy,
            message_info=message_info,
            bob_label_mi=bob_label_mi,
            alice_label_mi=alice_label_mi,
        )

        return s

    def intervention_input(self, game):
        mean_acc = 0.0
        scaler = 0.0

        for batch in self.dataset:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)
            sender_input, labels, receiver_input, aux_input = batch

            message = game.sender(sender_input, aux_input)
            # if Reinforce, agents return tuples
            if not self.is_gs:
                message = message[0]

            permutation = torch.randperm(receiver_input.size(0)).to(message.device)
            receiver_input = torch.index_select(receiver_input, 0, permutation)
            output = game.receiver(message, receiver_input, aux_input)

            if not self.is_gs:
                output = output[0]
            if self.is_gs and self.var_length:
                message = message.argmax(dim=-1)
                lengths = core.find_lengths(message)

                for i in range(lengths.size(0)):
                    curr_len = lengths[i]
                    _, rest = self.loss(
                        None,
                        None,
                        None,
                        output[i : i + 1, curr_len - 1],
                        labels[i : i + 1],
                        aux_input,
                    )
                    mean_acc += rest["acc"].item()
                    scaler += 1
            else:
                _, rest = self.loss(None, None, None, output, labels, aux_input)
                mean_acc += rest["acc"].mean().item()
                scaler += 1.0

        mean_acc /= scaler
        s = dict(
            mean_acc=mean_acc,
        )

        return s

    def on_epoch_end(self, _loss: float, _logs: core.Interaction, _epoch: int):
        game = self.trainer.game
        game.eval()

        intervantion_eval = self.intervention_message(game)
        validation_eval = self.validation(game)

        output = dict(
            epoch=self.epoch,
            intervention_message=intervantion_eval,
            validation=validation_eval,
        )
        if self.input_intervention:
            inp_intervention_eval = self.intervention_input(game)
            output.update(dict(input_intervention=inp_intervention_eval))

        output_json = json.dumps(output)
        print(output_json, flush=True)

        game.train()
        self.epoch += 1

    def validation(self, game):
        interactions = core.dump_interactions(
            game,
            self.dataset,
            gs=self.is_gs,
            device=self.device,
            variable_length=self.var_length,
        )

        messages = [interactions.message[i] for i in range(interactions.size)]
        entropy_messages = entropy(messages)
        labels = [interactions.labels[i] for i in range(interactions.size)]

        message_mapping = {}

        for message, label in zip(messages, labels):
            message = _hashable_tensor(message)
            label = _hashable_tensor(label)

            if message not in message_mapping:
                message_mapping[message] = {}

            message_mapping[message][label] = message_mapping[message].get(label, 0) + 1

        # majority vote per message
        correct = 0.0
        total = 0.0

        for labels in message_mapping.values():
            best_freq = None

            for freq in labels.values():
                if best_freq is None or freq > best_freq:
                    best_freq = freq

                total += freq
            correct += best_freq

        majority_accuracy = correct / total

        return dict(codewords_entropy=entropy_messages, majority_acc=majority_accuracy)
