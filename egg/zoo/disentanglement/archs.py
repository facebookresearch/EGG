# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import random
from collections import defaultdict


import egg.core as core
from torch.distributions import Categorical

class Receiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _):
        return self.fc(x)

class Sender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)

    def forward(self, x):
        x = self.fc1(x)
        return x

class SenderFFN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_output):
        super(SenderFFN, self).__init__()
        self.emb = nn.Linear(n_inputs, n_hidden, bias=False)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.emb(x)
        x = F.leaky_relu(x)
        x = self.fc(x)
        return x


class Receiver2(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x

class Shuffler(nn.Module):
    def __init__(self, sender):
        super().__init__()
        self.sender = sender

    def forward(self, x):
        messages, probs, entropies = self.sender(x)
        lengths = core.find_lengths(messages)

        for i in range(lengths.size(0)):
            l = lengths[i]
            z = torch.randperm(l)
            messages[i, :l] = messages[i, :l][z]
        return messages, probs, entropies


class PositionalSender(nn.Module):
    def __init__(self, n_attributes, n_values, vocab_size, max_len):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.vocab_size = vocab_size
        self.max_len = max_len

        log = 0
        k = 1

        while k < n_values:
            k *= vocab_size
            log += 1

        assert log * n_attributes <= max_len

        self.mapping = nn.Embedding(n_values, log)
        torch.nn.init.zeros_(self.mapping.weight)

        for i in range(n_values):
            value = i
            for k in range(log):
                self.mapping.weight[i, k] = value % vocab_size
                value = value // vocab_size

        assert (self.mapping.weight < vocab_size).all()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_attributes, self.n_values)
        x = x.argmax(dim=-1).view(batch_size * self.n_attributes)
        with torch.no_grad():
            x = self.mapping(x)
        x = x.view(batch_size, -1).long()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return x, zeros, zeros


class LinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc = nn.Linear(vocab_size * max_length, n_outputs)

        self.diagonal_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.eye_(self.diagonal_embedding.weight)

    def forward(self, x, *rest):
        with torch.no_grad():
            x = self.diagonal_embedding(x).view(x.size(0), -1)

        result = self.fc(x)

        zeros = torch.zeros(x.size(0), device=x.device)
        return result, zeros, zeros


class NonLinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, n_hidden, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc_1 = nn.Linear(vocab_size * max_length, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_outputs)

        self.diagonal_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.eye_(self.diagonal_embedding.weight)

    def forward(self, x, *rest):
        with torch.no_grad():
            x = self.diagonal_embedding(x).view(x.size(0), -1)

        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)

        zeros = torch.zeros(x.size(0), device=x.device)
        return x, zeros, zeros


class BosSender(nn.Module):
    def __init__(self, n_attributes, n_values, vocab_size, max_len):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.vocab_size = vocab_size
        self.max_len = max_len

        assert (n_attributes * n_values) <= max_len
        assert vocab_size >= n_attributes

        # each attribute uses unique symbol, avoiding zero
        self.attribute2symbol = [x for x in random.sample(range(vocab_size), n_attributes)]


    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, self.n_attributes, self.n_values)
        x = x.argmax(dim=-1).view(batch_size, self.n_attributes)

        result = torch.zeros(x.size(0), self.max_len, requires_grad=False).long()
        attribs = list(range(self.n_attributes))

        for i in range(x.size(0)):
            current_position = 0
            random.shuffle(attribs)

            for j in attribs:
                attr_value = x[i, j]
                result[i, current_position: current_position + attr_value] = self.attribute2symbol[j]
                current_position += attr_value
        result = result.to(x.device)

        """result = []
        attribs = list(range(self.n_attributes))

        for i in range(x.size(0)):
            current_position = 0
            random.shuffle(attribs)
            result.append([])

            for j in attribs:
                attr_value = x[i, j]
                result[-1].extend([self.attribute2symbol[j]] * attr_value.item())
            result[-1].extend([0] * (self.max_len - len(result[i])))

        result = torch.tensor(result, device=x.device)
        """
        zeros = torch.zeros(batch_size, result.size(1), device=x.device)
        return result, zeros, zeros


class FactorizedSender(nn.Module):
    def __init__(self, vocab_size, max_len, input_size, n_hidden):
        super().__init__()

        self.symbol_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, n_hidden),
                nn.LeakyReLU(),
                nn.Linear(n_hidden, vocab_size)
            ) for _ in range(max_len)
        ])

    def forward(self, bits):
        sequence = []
        log_probs = []
        entropy = []

        for generator in self.symbol_generators:
            logits = generator(bits.float())

            step_logits = F.log_softmax(logits, dim=-1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                sample = distr.sample()
            else:
                sample = step_logits.argmax(dim=-1)

            log_probs.append(distr.log_prob(sample))
            sequence.append(sample)

        zeros = torch.zeros(bits.size(0)).to(bits.device)

        #sequence.append(zeros.long())
        sequence = torch.stack(sequence).permute(1, 0)

        log_probs.append(zeros)
        log_probs = torch.stack(log_probs).permute(1, 0)

        entropy.append(zeros)
        entropy = torch.stack(entropy).permute(1, 0)

        return sequence, log_probs, entropy



class Freezer(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.eval()

    def train(self, mode):
        pass

    def forward(self, *input):
        with torch.no_grad():
            r = self.wrapped(*input)
        return r

class PlusOneWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *input):
        r1, r2, r3 = self.wrapped(*input)
        return r1 + 1, r2, r3


class PositionalDiscriminator(nn.Module):
    def __init__(self, vocab_size, n_hidden, embed_dim):
        super().__init__()
        self.encoder = core.RnnEncoder(vocab_size, embed_dim=embed_dim, n_hidden=n_hidden, cell='lstm')
        self.fc = nn.Linear(n_hidden, 2, bias=False)

    def forward(self, message):
        x = self.encoder(message)
        x = self.fc(x)
        return x


def _permute_dims(latent_sample):
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


class HistogramDiscriminator(nn.Module):
    def __init__(self, vocab_size, n_hidden, embed_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
                nn.Linear(vocab_size, n_hidden),
                nn.LeakyReLU(),
                nn.Linear(n_hidden, 2, bias=False)
                #nn.Linear(vocab_size, 2, bias=False)
                )

    def forward(self, histogram):
        #print(histogram)
        x = self.discriminator(histogram)
        return x


class SenderReceiverRnnReinforceWithDiscriminator(nn.Module):
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,
                 length_cost=0.0, discriminator=None, discriminator_weight=0.0, discriminator_transform=None):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

        self.discriminator = discriminator
        self.discriminator_weight = discriminator_weight
        self.discriminator_transform = discriminator_transform

    def forward(self, sender_input, labels, receiver_input=None):
        device = sender_input.device

        message, log_prob_s, entropy_s = self.sender(sender_input)

        message_lengths = core.find_lengths(message)
        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)

        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        policy_discriminator_loss = 0.0
        discriminator_loss = 0.0
        if self.discriminator is not None:
            with torch.no_grad():
                if self.discriminator_transform:
                    transformed = self.discriminator_transform(message)
                else:
                    transformed = message

                p_grammatical = self.discriminator(transformed).log_softmax(dim=-1)
                discriminator_loss = p_grammatical[:, 0] - p_grammatical[:, 1]
            policy_discriminator_loss = ((-discriminator_loss.detach() + self.mean_baseline['discriminator_loss']) * log_prob_s.sum(dim=-1)).mean()

        discriminator_train_loss = 0.0
        if self.discriminator is not None:
            positive_examples = message if self.discriminator_transform is None else self.discriminator_transform(message)
            negative_examples = _permute_dims(message) if self.discriminator_transform is None else _permute_dims(self.discriminator_transform(message))
            examples = torch.cat([positive_examples, negative_examples])
            batch_size = message.size(0)
            labels = torch.zeros(2 * batch_size, device=device).long()

            labels[batch_size:] = 1

            discriminator_predictions = self.discriminator(examples)
            discriminator_train_loss = F.cross_entropy(discriminator_predictions, labels)

            discriminator_train_acc = (discriminator_predictions.argmax(dim=-1) == labels).float().mean().item()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy - self.discriminator_weight * policy_discriminator_loss
        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean() + discriminator_train_loss

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)
            if self.discriminator:
                self.update_baseline('discriminator_loss', discriminator_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()
        if self.discriminator:
            rest['discriminator_loss'] = discriminator_loss.detach().mean().item()
            rest['discriminator_train_loss'] = discriminator_train_loss.detach().mean().item()
            rest['discriminator_train_acc'] = discriminator_train_acc

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class ReinforceWrapperFFN(nn.Module):
    def __init__(self, agent, length, vocab_size):
        super(ReinforceWrapperFFN, self).__init__()
        self.agent = agent
        self.length = length
        self.vocab_size = vocab_size

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)
        batch_size = logits.size(0)
        length_logits = logits.view(batch_size, self.length, self.vocab_size).permute(1,0,2)
        samples = []
        log_probs = []
        entropies = []
        for _length in length_logits:
            distr = Categorical(logits=_length)
            entropy = distr.entropy()

            if self.training:
                sample = distr.sample()
            else:
                sample = _length.argmax(dim=1)
            log_prob = distr.log_prob(sample)

            log_probs.append(log_prob.unsqueeze(1))
            entropies.append(entropy.unsqueeze(1))
            samples.append(sample.unsqueeze(1))

        return torch.cat(samples,1), torch.cat(log_probs,1), torch.cat(entropies,1)

if __name__ == '__main__':
    mapper = BosSender(n_attributes=3, n_values=3, vocab_size=10, max_len=15)
    input = torch.tensor([[0., 0., 1., 0., 1., 0., 1., 0., 0.]])
    print(mapper(input))
