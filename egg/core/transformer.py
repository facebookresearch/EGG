# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import find_lengths


class SinusoidalPositionEmbedding(nn.Module):
    """Implements sinusoidal positional embeddings"""

    def __init__(self, max_len: int, model_dim: int) -> None:
        super(SinusoidalPositionEmbedding, self).__init__()
        pos = torch.arange(0.0, max_len).unsqueeze(1).repeat(1, model_dim)
        dim = torch.arange(0.0, model_dim).unsqueeze(0).repeat(max_len, 1)
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / model_dim))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer("pe", pos.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Updates the input embedding with positional embedding
        Arguments:
            x {torch.Tensor} -- Input tensor
        Returns:
            torch.Tensor -- Input updated with positional embeddings
        """
        # fmt: off
        t = self.pe[:, :x.size(1), :]
        # fmt: on
        return x + t


class TransformerEncoder(nn.Module):
    """Implements a Transformer Encoder. The masking is done based on the positions of the <eos>
    token (with id 0).
    Two regimes are implemented:
    * 'causal' (left-to-right): the symbols are masked such that every symbol's embedding only can depend on the
        symbols to the left of it. The embedding of the <eos> symbol is taken as the representative.
    *  'non-causal': a special symbol <sos> is pre-pended to the input sequence, all symbols before <eos> are un-masked.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        num_layers: int = 1,
        positional_embedding=True,
        causal: bool = True,
    ) -> None:
        super().__init__()

        # in the non-causal case, we will use a special symbol prepended to the input messages which would have
        # term id of `vocab_size`. Hence we increase the vocab size and the max length
        if not causal:
            max_len += 1
            vocab_size += 1

        self.base_encoder = TransformerBaseEncoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_size=hidden_size,
            positional_embedding=positional_embedding,
        )
        self.max_len = max_len
        self.sos_id = torch.tensor([vocab_size - 1]).long()
        self.causal = causal

    def forward(
        self, message: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is None:
            lengths = find_lengths(message)

        batch_size = message.size(0)

        if not self.causal:
            prefix = self.sos_id.to(message.device).unsqueeze(0).expand((batch_size, 1))
            message = torch.cat([prefix, message], dim=1)
            lengths = lengths + 1

            max_len = message.size(1)
            len_indicators = (
                torch.arange(max_len).expand((batch_size, max_len)).to(lengths.device)
            )
            lengths_expanded = lengths.unsqueeze(1)
            padding_mask = len_indicators >= lengths_expanded

            transformed = self.base_encoder(message, padding_mask)
            # as the input to the agent, we take the embedding for the first symbol
            # which is always the special <sos> one
            transformed = transformed[:, 0, :]
        else:
            max_len = message.size(1)
            len_indicators = (
                torch.arange(max_len).expand((batch_size, max_len)).to(lengths.device)
            )
            lengths_expanded = lengths.unsqueeze(1)
            padding_mask = len_indicators >= lengths_expanded

            attn_mask = torch.triu(torch.ones(max_len, max_len).byte(), diagonal=1).to(
                lengths.device
            )
            attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float("-inf"))
            transformed = self.base_encoder(
                message, key_padding_mask=padding_mask, attn_mask=attn_mask
            )

            last_embeddings = []
            for i, l in enumerate(
                lengths.clamp(max=self.max_len - 1).cpu()
            ):  # noqa: E226
                last_embeddings.append(transformed[i, l, :])
            transformed = torch.stack(last_embeddings)

        return transformed


class TransformerBaseEncoder(torch.nn.Module):
    """
    Implements a basic Transformer Encoder module with fixed Sinusoidal embeddings.
    Initializations of the parameters are adopted from fairseq.
    Does not handle the masking w.r.t. message lengths, left-to-right order, etc.
    This is supposed to be done on a higher level.
    """

    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        num_layers,
        hidden_size,
        p_dropout=0.0,
        positional_embedding=True,
    ):
        super().__init__()

        # NB: they use a different one
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_dim = embed_dim
        self.max_source_positions = max_len
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = (
            SinusoidalPositionEmbedding(
                max_len + 1, embed_dim  # accounting for the forced EOS added by EGG
            )
            if positional_embedding
            else None
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim, num_heads=num_heads, hidden_size=hidden_size
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = p_dropout

        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.embed_dim ** -0.5)

    def forward(self, src_tokens, key_padding_mask=None, attn_mask=None):
        # embed tokens and positions
        x = self.embed_scale * self.embedding(src_tokens)

        if self.embed_positions is not None:
            x = self.embed_positions(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, key_padding_mask, attn_mask)

        x = self.layer_norm(x)

        #  T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_size,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, dropout=attention_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.normalize_before = True
        self.fc1 = torch.nn.Linear(self.embed_dim, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, self.embed_dim)
        # it seems there are two ways to apply layer norm - before (in tensor2tensor code)
        # or after (the original paper). We resort to the first as it is suggested to be more robust
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.init_parameters()

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _att = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x

    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)


class TransformerDecoder(torch.nn.Module):
    """
    Does not handle the masking w.r.t. message lengths, left-to-right order, etc.
    This is supposed to be done on a higher level.
    """

    def __init__(
        self, embed_dim, max_len, num_layers, num_heads, hidden_size, dropout=0.0
    ):
        super().__init__()

        self.dropout = dropout

        self.embed_positions = SinusoidalPositionEmbedding(max_len, embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(num_heads, embed_dim, hidden_size)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, embedded_input, encoder_out, key_mask=None, attn_mask=None):
        # embed positions
        embedded_input = self.embed_positions(embedded_input)

        x = F.dropout(embedded_input, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(x, encoder_out, key_mask=key_mask, attn_mask=attn_mask)

        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block. Follows an implementation in fairseq with args.decoder_normalize_before=True,
    i.e. order of operations is different from those in the original paper.
    """

    def __init__(
        self,
        num_heads,
        embed_dim,
        hidden_size,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, dropout=attention_dropout
        )  # self-attn?

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        # NB: we pass encoder state as a single vector at the moment (form the user-defined module)
        # hence this attention layer is somewhat degenerate/redundant. Nonetherless, we'll have it
        # for (a) proper compatibility (b) in case we'll decide to pass multipel states
        self.encoder_attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, dropout=attention_dropout
        )

        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.fc1 = torch.nn.Linear(self.embed_dim, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, self.embed_dim)

        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x, encoder_out, key_mask=None, attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=key_mask, attn_mask=attn_mask
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.encoder_attn_layer_norm(x)
        # would be a single vector, so no point in attention at all
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm(x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        return x, attn
