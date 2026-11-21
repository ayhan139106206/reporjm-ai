import torch
from torch import nn

class Monster1(nn.Module):
    def __init__(self, vocab=4096, dim=256, layers=4, heads=4):
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(1024, dim)
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token(x) + self.position(pos)
        mask = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device),
            diagonal=1
        ).bool()
        h = self.transformer(h, mask=mask)
        return self.head(self.norm(h))
