import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


class GNNConv(nn.Module):
    def __init__(self, emb_dim):
        super(GNNConv, self).__init__()
        """
        emb_dim: int
        """

        ### Edge feature encoder
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)

        ### Neighbourhood Message Function
        self.gnn_msg = fn.v_mul_e(lhs_field='h', rhs_field='he', out='m')

        ### Neighbourhood Aggregator Function
        self.gnn_agg = fn.sum(msg='m', out='h')

        ### State transformer
        self.linear3 = nn.Linear(emb_dim, emb_dim)
        self.linear4 = nn.Linear(emb_dim, emb_dim)

    def forward(self, g, h, he):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = h

            ### Transformation of edge feature vector
            he = F.relu(self.linear1(he))
            he = self.linear2(he)
            g.edata['he'] = he

            ### Neighbourhood aggregation
            g.update_all(self.gnn_msg, self.gnn_agg)

            ### Transformation of state vector
            h = g.ndata['h']
            h = F.relu(self.linear3(h))
            h = self.linear4(h)

            return h


class DGQN(nn.Module):
    def __init__(self, in_dim, act_dim, emb_dim, num_layers):
        super(DGQN, self).__init__()
        """
        emb_dim: int
        num_layers: int
        """
        self.emb_dim = emb_dim

        ### Edge initial MLP
        self.linear1 = nn.Linear(in_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)

        ### List of graph conv layers
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GNNConv(emb_dim))

        ### Graph readout MLP
        self.linear3 = nn.Linear(emb_dim, emb_dim)
        self.linear4 = nn.Linear(emb_dim, act_dim)
        

    def forward(self, g):
        with g.local_scope():

            ### Transform the observation on the edges
            he = g.edata['obs']
            he = F.relu(self.linear1(he))
            he = self.linear2(he)

            ### Set inital node feature vectors all the same
            h = torch.ones(g.num_nodes(), self.emb_dim)

            ### Message passing among graph nodes
            for layer in range(self.num_layers):
                h = self.convs[layer](g, h, he)
                h = F.relu(h)

            ### Graph Readout
            g.ndata['h'] = h
            hg = dgl.sum_nodes(g, 'h')
            hg = F.relu(self.linear3(hg))
            hg = self.linear4(hg)
            return hg


