# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from egg import core
from egg.zoo.compo_vs_generalization.archs import PlusOneWrapper, Receiver, Sender


class BaseEncoder(nn.Module):
    """encoder used for both the ModifSender and the ModifReceiver"""

    def __init__(self, input_size, hidden_size, device="cpu"):
        super(BaseEncoder, self).__init__()
        self.hid_dim = hidden_size
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.sem_embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        bs = input.size(0)
        hidden = self.init_hidden(bs)
        embedded = self.embedding(input)
        embedded = embedded.view(-1, bs, self.hid_dim)
        output, hidden = self.gru(
            embedded, hidden
        )  # [seq_len, bs, hid_dim(*2 if bidirectional)]
        output = output.transpose(1, 0)  # [bs, seq_len, hid_dim(*2 if bidirectional)]
        sem_embs = self.sem_embedding(input)

        return output, hidden, sem_embs

    def init_hidden(self, bs):
        return torch.zeros(1, bs, self.hid_dim, device=self.device)


class AttnMasked(nn.Module):
    """
    implementation taken from B.Lake's meta_seq2seq code:
    https://github.com/facebookresearch/meta_seq2seq/blob/59c3b4aafebf387bcd4e45626d8d91b66e6e5dff/model.py#L223
    """

    def __init__(self):
        super(AttnMasked, self).__init__()

    def forward(self, Q, K, V, key_length_mask):
        #
        # Input
        #  Q : Matrix of queries; batch_size x n_queries x query_dim
        #  K : Matrix of keys; batch_size x n_memory x query_dim
        #  V : Matrix of values; batch_size x n_memory x value_dim
        #  key_length_mask: mask telling me which positions to ignore (True)
        #    and which to consider (False),
        #    the True/False assignment is given by the torch.masked_fill_,
        #    this method fills in a value for each True position;
        #    batch_size x

        # Output
        #  R : soft-retrieval of values; batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; batch_size x n_queries x n_memory
        query_dim = torch.tensor(float(Q.size(2)))
        if Q.is_cuda:
            query_dim = query_dim.cuda()
        attn_weights = torch.bmm(
            Q, K.transpose(1, 2)
        )  # batch_size x n_queries x n_memory
        attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
        attn_weights.masked_fill_(key_length_mask, -1000)
        attn_weights = F.softmax(
            attn_weights, dim=2
        )  # batch_size x n_queries x n_memory
        R = torch.bmm(attn_weights, V)  # batch_size x n_queries x value_dim
        return R, attn_weights


class SenderDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, max_length, device):
        super(SenderDecoder, self).__init__()
        self.hid_dim = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size + 3, self.hid_dim)
        # +3 for [fake EOS emb, fake SOS emb, fake EOS semantic emb]
        self.gru = nn.GRU(self.hid_dim, self.hid_dim)
        self.attn = AttnMasked()
        self.out = nn.Linear(self.hid_dim, self.output_size)

    def get_per_step_logits(self, input, hidden, keys, values, length_mask):
        embedded = self.embedding(input)  # [1, bs, hid_dim]
        output, hidden = self.gru(
            embedded, hidden
        )  # [1, bs, hid_dim], [1, bs, hid_dim]
        # Attention
        queries = output.transpose(1, 0)  # [bs, 1, hid_dim]
        weighted_values, weights = self.attn(
            queries, keys, values, length_mask
        )  # [bs, 1, hid_dim], [bs, 1, max_len]

        logits = self.out(weighted_values[:, 0] + output[0])
        return logits, hidden, weights[:, 0, :]

    def init_batch(self, encoder_outputs, sem_embs):
        bs = encoder_outputs.size(0)
        input = (
            torch.zeros(1, bs, dtype=torch.long, device=self.device)
            + self.output_size
            + 2
        )  # [1, bs]
        hidden = self.init_hidden(bs)  # [1, bs, hid_dim]

        fake_EOS = self.embedding(
            torch.zeros(bs, 1, dtype=torch.long, device=self.device) + self.output_size
        )
        fake_EOS_sem = self.embedding(
            torch.zeros(bs, 1, dtype=torch.long, device=self.device)
            + self.output_size
            + 1
        )
        keys = torch.cat([encoder_outputs, fake_EOS], dim=1)  # [bs, max_len, hid_dim]
        values = torch.cat([sem_embs, fake_EOS_sem], dim=1)  # [bs, max_len, hid_dim]
        length_mask = torch.zeros(
            bs, 1, keys.shape[1], dtype=torch.bool, device=self.device
        )

        return bs, input, hidden, keys, values, length_mask

    def forward(self, encoder_outputs, sem_embs, deterministic=False):
        bs, input, hidden, keys, values, length_mask = self.init_batch(
            encoder_outputs, sem_embs
        )
        sequence, per_step_logits, entropy, attn_weights = [], [], [], []
        for _ in range(self.max_length):
            logits, hidden, weights = self.get_per_step_logits(
                input, hidden, keys, values, length_mask
            )
            distr = Categorical(logits=logits)

            entropy.append(distr.entropy())
            if self.training:
                x = distr.sample()
            else:
                x = logits.argmax(dim=-1)
            per_step_logits.append(distr.log_prob(x))

            sequence.append(x) if not deterministic else sequence.append(logits)
            input = x[None]
            attn_weights.append(weights)
        zeros = torch.zeros((bs, 1), device=self.device)

        sequence = torch.stack(sequence, 1)  # [bs, max_len, out_dim]
        if not deterministic:
            sequence = torch.cat(
                [sequence, zeros.long()], dim=-1
            )  # [bs, max_len + 1, out_dim]
        per_step_logits = torch.cat(
            [torch.stack(per_step_logits, 1), zeros], dim=-1
        )  # [bs, max_len + 1, out_dim]
        entropy = torch.cat(
            [torch.stack(entropy, 1), zeros], dim=-1
        )  # [bs, max_len + 1, out_dim]
        attn_weights = torch.stack(attn_weights, 1)

        return sequence, per_step_logits, entropy, attn_weights

    def init_hidden(self, bs):
        return torch.zeros(1, bs, self.hid_dim, device=self.device)


class ReceiverDecoder(SenderDecoder):
    def init_batch(self, encoder_outputs, sem_embs):
        bs = encoder_outputs.size(0)
        input = (
            torch.zeros(1, bs, dtype=torch.long, device=self.device)
            + self.output_size
            + 2
        )  # [1, bs]
        hidden = self.init_hidden(bs)  # [1, bs, hid_dim]
        keys = encoder_outputs  # [bs, max_len, hid_dim]
        values = sem_embs  # [bs, max_len, hid_dim]
        length_mask = torch.zeros(
            bs, 1, keys.shape[1], dtype=torch.bool, device=self.device
        )

        return bs, input, hidden, keys, values, length_mask

    def forward(self, encoder_outputs, sem_embs, lengths, force=None):

        bs, input, hidden, keys, values, length_mask = self.init_batch(
            encoder_outputs, sem_embs
        )
        # we want to ignore the previous dummy length_mask
        # length_mask is TRUE for positions which are to be IGNORED
        length_mask = (
            torch.arange(encoder_outputs.size(1), device=self.device)[None, :]
            >= lengths[:, None]
        )
        length_mask = length_mask[:, None, :]  # [bs, 1, max_message_length]

        per_step_logits, attn_weights = [], []
        for i in range(self.max_length):
            logits, hidden, weights = self.get_per_step_logits(
                input, hidden, keys, values, length_mask
            )
            per_step_logits.append(logits)
            attn_weights.append(weights)
            input = logits.argmax(dim=-1) if force is None else force[:, i]
            input = input[None]

        per_step_logits = torch.stack(per_step_logits, 1)  # [bs, n_attrs, n_values]
        attn_weights = torch.stack(attn_weights, 1)

        top_logits_ = entropy_ = torch.zeros(bs, device=self.device)
        return per_step_logits, top_logits_, entropy_, attn_weights


class ModifSender(nn.Module):
    """enc-dec architecture implementing the sender in the communication game"""

    def __init__(self, opts):
        super(ModifSender, self).__init__()
        self.encoder = BaseEncoder(opts.n_values, opts.hidden, device=opts.device)
        self.decoder = SenderDecoder(
            opts.vocab_size + 1, opts.hidden, opts.max_len, device=opts.device
        )

        self.n_attributes = opts.n_attributes
        self.n_values = opts.n_values

    def forward(self, x, aux_input=None, deterministic=False):
        # change the egg-style format (concatenation of one-hot encodings) to `ordinary`
        #   input format (vector of indices):
        bs = x.size(0)
        x = (
            x.view(bs * self.n_attributes, self.n_values)
            .nonzero()[:, 1]
            .view(bs, self.n_attributes)
        )

        enc_output, hidden, sem_embs = self.encoder(x)
        sequence, top_logits, entropy, attn_weights = self.decoder(
            enc_output, sem_embs, deterministic=deterministic
        )
        return sequence, top_logits, entropy


class ModifReceiver(nn.Module):
    """enc-dec architecture implementing the receiver in the communication game"""

    def __init__(self, opts):
        super(ModifReceiver, self).__init__()
        self.encoder = BaseEncoder(opts.vocab_size + 1, opts.hidden, device=opts.device)
        self.decoder = ReceiverDecoder(
            opts.n_values, opts.hidden, opts.n_attributes, device=opts.device
        )

        self.n_attributes = opts.n_attributes
        self.n_values = opts.n_values

    def forward(self, message, input=None, aux_input=None, lengths=None):
        enc_output, hidden, sem_embs = self.encoder(message)
        if lengths is None:
            lengths = core.find_lengths(message)
        per_step_logits, top_logits_, entropy_, attn_weights = self.decoder(
            enc_output, sem_embs, lengths
        )
        per_step_logits = per_step_logits.view(-1, self.n_attributes * self.n_values)
        return per_step_logits, top_logits_, entropy_


class OrigSender(nn.Module):
    """mimicking the architecture used in `compo_vs_generalization/train.py`"""

    def __init__(self, opts):
        super(OrigSender, self).__init__()
        n_dim = opts.n_attributes * opts.n_values
        sender = Sender(n_inputs=n_dim, n_hidden=opts.hidden)
        sender = core.RnnSenderReinforce(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb,
            hidden_size=opts.hidden,
            max_len=opts.max_len,
            cell="gru",
        )
        self.sender = PlusOneWrapper(sender)

    def forward(self, *input, **kwargs):
        return self.sender(*input, **kwargs)


class OrigReceiver(nn.Module):
    """mimicking the architecture used in `compo_vs_generalization/train.py`"""

    def __init__(self, opts):
        super(OrigReceiver, self).__init__()
        n_dim = opts.n_attributes * opts.n_values
        receiver = Receiver(n_hidden=opts.hidden, n_outputs=n_dim)
        self.receiver = core.RnnReceiverDeterministic(
            receiver,
            opts.vocab_size + 1,
            opts.receiver_emb,
            opts.hidden,
            cell="gru",
        )

    def forward(self, *input):
        return self.receiver(*input)


class RnnSenderDeterministic(core.RnnSenderReinforce):
    """
    a modification of `core.RnnSenderReinforce`

    `forward()` is modified so that it returns logits of candidates at each position.
    (in order to train the model using regular SGD)
    """

    def forward(self, x, aux_input=None):
        prev_hidden = [self.agent(x, aux_input)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        prev_c = [
            torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)
        ]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        # logits = []
        # entropy = []

        for _ in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                prev_hidden[i] = h_t
                input = h_t

            logits = self.hidden_to_output(h_t)
            step_logits = F.log_softmax(logits, dim=1)
            distr = Categorical(logits=step_logits)

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            input = self.embedding(x)
            sequence.append(logits)

        sequence = torch.stack(sequence, 1)
        return sequence, None, None


class OrigSenderDeterministic(nn.Module):
    """OrigSender changed for learning alone experiments (training with regular SGD)"""

    def __init__(self, opts):
        super(OrigSenderDeterministic, self).__init__()
        n_dim = opts.n_attributes * opts.n_values
        sender = Sender(n_inputs=n_dim, n_hidden=opts.hidden)
        self.sender = RnnSenderDeterministic(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb,
            hidden_size=opts.hidden,
            max_len=opts.max_len,
            cell="gru",
        )

    def forward(self, *input, **kwargs):
        return self.sender(*input, **kwargs)
