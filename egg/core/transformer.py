import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, seqlen, dmodel):
        super(SinusoidalPositionEmbedding, self).__init__()
        pos = torch.arange(0., seqlen).unsqueeze(1).repeat(1, dmodel)
        dim = torch.arange(0., dmodel).unsqueeze(0).repeat(seqlen, 1)
        div = torch.exp(- math.log(10000) * (2*(dim//2)/dmodel))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer('pe', pos.unsqueeze(0))

    def forward(self, x):
        t = self.pe[:, :x.size(1), :]
        return x + t


class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, n_layers, hidden_size,
                 positional_embedding=True):
        super().__init__()

        #self.dropout = 0.0
        # NB: they use a different one
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.padding_idx = 0
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

        #self.normalize = False
        #if self.normalize:
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, src_tokens, encoder_padding_mask=None):
        #print(encoder_padding_mask)
        # embed tokens and positions
        x = self.embed_scale * self.embedding(src_tokens)

        if self.embed_positions is not None:
            x = self.embed_positions(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        #encoder_padding_mask = src_tokens.eq(self.padding_idx)
        #if not encoder_padding_mask.any():
        #    encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        #if self.normalize:
        x = self.layer_norm(x)

        #  T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder_ffn_embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=0.0)
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        #self.dropout = 0.0
        #self.activation_dropout = 0.0

        self.normalize_before = True
        self.fc1 = torch.nn.Linear(self.embed_dim, encoder_ffn_embed_dim) #Linear(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = torch.nn.Linear(encoder_ffn_embed_dim, self.embed_dim) #Linear(encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


class Model(torch.nn.Module):
    def __init__(self, enc, emb_dim, out_dim):
        super().__init__()

        self.enc = enc
        self.fc = torch.nn.Linear(emb_dim, out_dim)

    def forward(self, x):
        x = self.enc(x)
        x = self.fc(x[:, -1, :])
        return x


if __name__ == '__main__':
    torch.manual_seed(7)

    encoder = TransformerEncoder(vocab_size=3, max_len=4, embed_dim=10, num_heads=1, n_layers=1,
                                 hidden_size=10, positional_embedding=True)
    model = Model(encoder, emb_dim=10, out_dim=2)

    BATCH_X = torch.tensor([[1, 1, 1],
                            [1, 0, 0],
                            [1, 0, 1],
                            [1, 1, 0],

                            [0, 0, 0],
                            [0, 1, 1],
                            [0, 0, 0],
                            [0, 1, 0]
                            ]).long() + 1
    # 0 is padding

    #BATCH_Y = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1]).long()
    BATCH_Y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]).long()

    optim = torch.optim.Adam(params=model.parameters())

    for i in range(1000):
        optim.zero_grad()

        output = model(BATCH_X)
        loss = F.cross_entropy(output, BATCH_Y)
        loss.backward()
        optim.step()

        if i % 50 == 0:
            acc = (output.argmax(dim=1) == BATCH_Y).float().mean()
            print(f'acc = {acc}')

