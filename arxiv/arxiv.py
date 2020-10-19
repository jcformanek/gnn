import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
from dgl.nn import GraphConv, SAGEConv
import dgl.function as fn

import ogb
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


### Load the Arxiv dataset
def load_dataset():
    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()

    # there is only one graph in Node Property Prediction datasets
    g, labels = dataset[0]
    g = dgl.add_self_loop(g)
    
    return g, labels, split_idx


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(input_dim, hidden_dim))
        for i in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
        self.convs.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g):
        with g.local_scope():
            h = g.ndata['feat']
            for conv in self.convs[:-1]:
                h = conv(g, h)
                h = F.relu(h)
            h = self.convs[-1](g, h)
            return h


def train(model, g, labels, split_idx, optimizer, loss_func):
    model.train()
    optimizer.zero_grad()
    pred = model(g)[split_idx['train']]
    loss = loss_func(pred, labels[split_idx['train']].view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval(model, g, labels, split_idx, evaluator):
    model.eval()

    pred = model(g)
    y_pred = pred.argmax(dim=1).view(-1, 1)

    train_perf = evaluator.eval({
        'y_true': labels[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })

    valid_perf = evaluator.eval({
        'y_true': labels[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })

    test_perf = evaluator.eval({
        'y_true': labels[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })

    return train_perf, valid_perf, test_perf


def run_experiment(num_epochs=100, lr=0.01, hidden_dim=256, num_layers=3):
    g, labels, split_idx = load_dataset()
    evaluator = Evaluator(name='ogbn-arxiv')
    loss_func = nn.CrossEntropyLoss()
    model = GCN(128, hidden_dim, 40, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print("=====Epoch {}=====".format(epoch + 1))
        loss = train(model, g, labels, split_idx, optimizer, loss_func)
        train_perf, valid_perf, test_perf = eval(model, g, labels, split_idx, evaluator)
        print('Loss:', loss)
        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        print()


if __name__ == "__main__":
    run_experiment()