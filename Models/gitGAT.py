import torch
import torch.nn as nn

class GAT(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.fciandj = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.key = nn.Linear(2 * self.input_dim, 1, bias=False)

    def forward(self, h, hjs, n_list):
        wh = self.fciandj(h)
        whjs = self.fciandj(hjs)
        temp = (torch.empty((0, wh.shape[1]))).to(self.device)
        n_list = n_list.tolist()
        for ind in range(len(h)):
            wh_i = wh[ind, :]
            temp = torch.concat([temp, wh_i.repeat(n_list[ind], 1)])
        qk = torch.concat([temp, whjs], dim=1)
        wqk = self.key(qk)
        middlevalue = nn.functional.leaky_relu(wqk)
        eis = torch.split(middlevalue, n_list)
        whis = torch.split(whjs, n_list)
        new_h = (torch.empty((self.input_dim, 0))).to(self.device)
        for ind in range(len(eis)):
            att_w_ind = nn.functional.softmax(eis[ind], dim=0)
            mul = att_w_ind * whis[ind]
            new_i = torch.sum(mul, dim=0).unsqueeze(-1)
            new_h = torch.concat([new_h, new_i], dim=1)
        return nn.functional.relu(new_h.T)


class Multihead_GaAtN(nn.Module):
    def __init__(self, n_heads, m, input_dim, out_dim, device):
        super().__init__()
        self.m = m
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.device = device
        self.MHGAT = nn.ModuleList()
        for _ in range(self.n_heads):
            self.MHGAT.append(GAT(self.input_dim, self.device))
        self.w0 = nn.Linear(self.input_dim*self.n_heads, self.input_dim, bias=False)
        self.getgate = GetGate(self.n_heads, self.input_dim, self.device)


    def forward(self, h, adj, n_list):

        g = self.getgate(h) #num of residue, n-heads
        hjs = (torch.empty((0, h.shape[1]))).to(self.device)
        for i in range(len(adj)):
            hjs = torch.concat([hjs, h[adj[i, :].bool()]])
        heads = (torch.zeros(len(h), 0)).to(self.device)
        for n_head, mhgat in enumerate(self.MHGAT):
            new_h_per_head = mhgat(h, hjs, n_list)
            gh = g[:, n_head].unsqueeze(-1)
            h_head = gh * new_h_per_head
            heads = torch.concat([heads, h_head], dim=1)
        return self.w0(heads)

class Multihead_GAT(nn.Module):
    def __init__(self, n_heads, m, input_dim, out_dim, device):
        super().__init__()
        self.m = m
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.device = device
        self.MHGAT = nn.ModuleList()
        for _ in range(self.n_heads):
            self.MHGAT.append(GAT(self.input_dim, self.device))
        self.w0 = nn.Linear(self.input_dim*self.n_heads, self.input_dim, bias=False)


    def forward(self, h, adj, n_list):

        hjs = (torch.empty((0, h.shape[1]))).to(self.device)
        for i in range(len(adj)):
            hjs = torch.concat([hjs, h[adj[i, :].bool()]])
        heads = (torch.zeros(len(h), 0)).to(self.device)
        for n_head, mhgat in enumerate(self.MHGAT):
            new_h_per_head = mhgat(h, hjs, n_list)
            heads = torch.concat([heads, new_h_per_head], dim=1)
        return self.w0(heads)
