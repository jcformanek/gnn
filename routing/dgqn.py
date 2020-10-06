import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


class ReplayBuffer():

    def __init__(self, size=10000):
        self.obs_mem = []
        self.act_mem = []
        self.rew_mem = []
        self.next_obs_mem = []
        self.ctr = 0
        self.size = size

    def store(self, obs, act, rew, next_obs):
        if self.ctr < self.size:
            self.obs_mem.append(obs)
            self.act_mem.append(act)
            self.rew_mem.append(rew)
            self.next_obs_mem.append(next_obs)
        else:
            idx = self.ctr % self.size
            self.obs_mem[idx] = obs
            self.act_mem[idx] = act
            self.rew_mem[idx] = rew
            self.next_obs_mem[idx] = next_obs

        self.ctr += 1

    def sample(self, num_samples):
        if self.ctr < num_samples:
            raise ValueError("Too few entries in replay buffer to sample!")
        
        max_idx = min(self.ctr, self.size)
        idxs = random.sample(range(max_idx), num_samples)

        obs_sample = [self.obs_mem[i] for i in idxs]
        act_sample = [self.act_mem[i] for i in idxs]
        rew_sample = [self.rew_mem[i] for i in idxs]
        next_obs_sample = [self.next_obs_mem[i] for i in idxs]

        return obs_sample, act_sample, rew_sample, next_obs_sample


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
    def __init__(self, num_routes, emb_dim, num_layers):
        super(DGQN, self).__init__()
        """
        emb_dim: int
        num_layers: int
        """
        self.emb_dim = emb_dim

        ### Edge initial MLP
        self.linear1 = nn.Linear(num_routes + 1, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)

        ### List of graph conv layers
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GNNConv(emb_dim))

        ### Graph readout MLP
        self.linear3 = nn.Linear(emb_dim, emb_dim)
        self.linear4 = nn.Linear(emb_dim, num_routes)
        

    def forward(self, g, obs):
        with g.local_scope():

            ### Transform the observation on the edges
            he = F.relu(self.linear1(obs))
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


