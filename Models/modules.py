import torch.nn as nn

class pepSeq(nn.Module):
    def __init__(self, input, hidden, mode, n_heads):
        super().__init__()
        self.mode = mode
        self.input = input
        self.heads = n_heads
        self.Attention = nn.MultiheadAttention(self.input, num_heads=self.heads)
        self.norm1 = nn.LayerNorm(self.input)
        self.FF = nn.Sequential(nn.Linear(in_features=self.input, out_features=self.input*2),
                                nn.ReLU(),
                                nn.Linear(in_features=self.input*2, out_features=self.input))
        self.norm2 = nn.LayerNorm(self.input)
    def forward(self, seqEmb):
        att, _ = self.Attention(seqEmb, seqEmb, seqEmb)
        out = self.norm1(seqEmb+att)
        outfeat = self.norm2(self.FF(out)+out)
        return outfeat

from newGATs import *
class GATEncoder(nn.Module):
    def __init__(self, n_heads, m, input_dim, hidden_size, device):
        super().__init__()
        self.m = m
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.device = device
        self.gan = Multihead_GAT(self.n_heads, self.m, self.input_dim, self.input_dim, self.n_heads*self.input_dim, self.device)
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.ff = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=self.input_dim*2),
                                nn.ReLU(),
                                nn.Linear(in_features=self.input_dim*2, out_features=self.input_dim))
        self.norm2 = nn.LayerNorm(self.input_dim)

    def forward(self, h, adj, n_list):
        att = self.gan(h, adj, n_list)
        t = self.norm1(att+h)
        ff = self.ff(t)
        encode_h = self.norm2(ff+t)
        return encode_h
