import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, seqlen, dmodel):
        super(SinusoidalPositionEmbedding, self).__init__()
        pos = torch.arange(0., seqlen).unsqueeze(1).repeat(1, dmodel)
        dim = torch.arange(0., dmodel).unsqueeze(0).repeat(seqlen, 1)
        div = torch.exp(- math.log(10000) * (2 * (dim // 2) / dmodel))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer('pe', pos.unsqueeze(0))

    def forward(self, x):
        t = self.pe[:, :x.size(1), :]
        return x + t


class TransformerEncoder(torch.nn.Module):
    """
    Implements a basic Transformer Encoder module with fixed Sinusoidal embeddings.
    Initializations of the parameters are adopted from fairseq.
    """

    def __init__(self, vocab_size, max_len, embed_dim, num_heads, n_layers, hidden_size,
                 p_dropout=0.0,
                 positional_embedding=True):
        super().__init__()

        # NB: they use a different one
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_dim = embed_dim
        self.max_source_positions = max_len
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionEmbedding(max_len, embed_dim) if positional_embedding else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                encoder_ffn_embed_dim=hidden_size
            )
            for _ in range(n_layers)
        ])
        self.dropout = p_dropout

        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.embed_dim ** -0.5)

    def forward(self, src_tokens, encoder_padding_mask=None):
        # embed tokens and positions
        x = self.embed_scale * self.embedding(src_tokens)

        if self.embed_positions is not None:
            x = self.embed_positions(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        x = self.layer_norm(x)

        #  T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder_ffn_embed_dim, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads,
                                                     dropout=attention_dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.dropout = dropout

        self.normalize_before = True
        self.fc1 = torch.nn.Linear(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = torch.nn.Linear(encoder_ffn_embed_dim, self.embed_dim)
        # it seems there are two ways to apply layer norm - before (in tensor2tensor code)
        # or after (the original paper). We resort to the first as it is suggested to be more robust
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.init_parameters()

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _att = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        # fairseq has an activation dropout here
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x

    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.)


class TransformerDecoder(torch.nn.Module):
    def __init__(self, embed_dim, max_len, n_decoder_layers,
                 attention_heads, ffn_embed_dim, dropout=0.0):
        super().__init__()

        self.dropout = dropout

        self.embed_positions = SinusoidalPositionEmbedding(max_len, embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(attention_heads, embed_dim, ffn_embed_dim)
            for _ in range(n_decoder_layers)
        ])

        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, embedded_input, encoder_out, self_attn_mask):
        # embed positions
        x = self.embed_positions(embedded_input)

        x = F.dropout(embedded_input, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(x, encoder_out, self_attn_mask=self_attn_mask)

        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, decoder_attention_heads, decoder_embed_dim, decoder_ffn_embed_dim, attention_dropout=0.0):
        super().__init__()

        self.embed_dim = decoder_embed_dim
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=attention_dropout
        )  # self-attn?

        self.dropout = 0.0
        self.activation_dropout = 0.0
        self.normalize_before = True

        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.encoder_attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=0.0)

        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.fc1 = torch.nn.Linear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = torch.nn.Linear(decoder_ffn_embed_dim, self.embed_dim)

        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(self, x,
                encoder_out,
                self_attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_mask)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.encoder_attn_layer_norm(x)
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
"""
if __name__ == '__main__':
    max_len = 4

    src = torch.eye(8).long()
    state = torch.randn(size=(8, 10))

    model = TransformerDecoder(vocab_size=2, embed_dim=10, decoder_output_dim=10,
                               input_embed_dim=10, max_len=8,
                               n_decoder_layers=2, attention_heads=2, ffn_embed_dim=10)

    mask = src.new(max_len, max_len).float()
    mask.fill_(float('-inf'))
    print(mask)
    mask = torch.triu(mask, 1)
    print(mask)

    model(src, state, mask[:, 0])
"""

class Model(torch.nn.Module):
    def __init__(self, enc, emb_dim, out_dim):
        super().__init__()

        self.enc = enc
        self.fc = torch.nn.Linear(emb_dim, out_dim)

    def forward(self, x, mask):
        x = self.enc(x, mask)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == '__main__':
    torch.manual_seed(7)

    encoder = TransformerEncoder(vocab_size=3, max_len=4, embed_dim=10, num_heads=1, n_layers=1,
                                 hidden_size=10, positional_embedding=True)

    decoder = TransformerDecoder(vocab_size=4, embed_dim=5, max_len=4, n_decoder_layers=1,
                                 attention_heads=1, ffn_embed_dim=10, decoder_output_dim=10)

    """BATCH_X = torch.tensor([[1, 1, 1, -1],
                            [1, 0, 0, -1],
                            [1, 0, 1, -1],
                            [1, 1, 0, -1],

                            [0, 0, 0, -1],
                            [0, 1, 1, -1],
                            [0, 0, 0, -1],
                            [0, 1, 0, -1]
                            ]).long() + 1
    # 0 is padding
    #padding_mask = torch.zeros_like(BATCH_X).byte()
    #padding_mask[:, -1] = 1
    #print(padding_mask, 'orig')

    BATCH_Y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]).long()


    print(decoder(BATCH_X[:, 0], BATCH_X, None))

    """
    state = torch.randn(size=(1, 5))

    self_inp = torch.zeros(size=(1, 4)).long()
    mask = torch.ones_like(self_inp).byte()
    mask[:, 0] = 0
    mask[:, 1] = 0

    decoder(self_inp, state, mask)

    """
    optim = torch.optim.Adam(params=model.parameters())

    for i in range(1000):
        optim.zero_grad()

        output = model(BATCH_X, padding_mask)
        loss = F.cross_entropy(output, BATCH_Y)
        loss.backward()
        optim.step()

        if i % 50 == 0:
            acc = (output.argmax(dim=1) == BATCH_Y).float().mean()
            print(f'acc = {acc}')
            
    """

