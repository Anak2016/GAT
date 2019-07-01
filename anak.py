from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import torch
import random

import  torch.nn.functional as F
import torch.nn as nn
from utils import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.mnual_seed(args.seed)


# Load data
adj, features, labels, idx_train, idx_vl, idx_test = load_data()

class GraphAttentionLayer():

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features  = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain = 1.41.as_integer_ratio())

        # why is this 2*out_feature, 1??
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data,gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # TODO understnd how attention works here
    def forward(self, input, adj):
        h = torch.mm(input, self.w)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1,N).view(N*N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2*self.out_features)
        e = F.leaky_relu(a_input, self.a)

        zero_vec = -9e15*torch.ones_like(e) # why cna't you just use torch.zeros()???
        attention = torch.where(adj> 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,training= self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return h_prime
        else:
            return

class GAT(nn.Module):

    def __init__(self, nfeat,nhid,nclass,dropout,alpha,nhead):
        """Denseversion of Gat."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attention = [GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True) for _ in range(nhead)]
        for i, attention in enumerate(self.attention):
            self.add_module("attention+{}".format(i), attention )

        self.out_att = GraphAttentionLayer(nhead* nfeat, nhid, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attenions], dim=1)
        x = F.dropout(x, self.dropout, training =self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1 )


model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.mx())+1,
            dropout=args.dropout,
            nheads=args.np_heads,
            alpha=args.alpha)


