import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
from dgl.nn import GraphConv, SAGEConv, GATConv, GINConv
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


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, gnn_type):
        super(GNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        self.convs = nn.ModuleList()

        if gnn_type == "gcn":
            self.convs.append(GraphConv(input_dim, hidden_dim))
        elif gnn_type == "sage":
            self.convs.append(SAGEConv(input_dim, hidden_dim, "gcn"))
        elif gnn_type == "gat":
            self.convs.append(GATConv(input_dim, hidden_dim, num_heads=3))
        else:
            raise ValueError("Invalid gnn_type")

        for i in range(num_layers - 2):
            if gnn_type == "gcn":
                self.convs.append(GraphConv(hidden_dim, hidden_dim))
            elif gnn_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, "gcn"))
            elif gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim, num_heads=3))

        if gnn_type == "gcn":
            self.convs.append(GraphConv(hidden_dim, output_dim))
        elif gnn_type == "sage":
            self.convs.append(SAGEConv(hidden_dim, output_dim, "gcn"))
        elif gnn_type == "gat":
            self.convs.append(GATConv(hidden_dim, output_dim, num_heads=3))
        

    def forward(self, g):
        with g.local_scope():
            h = g.ndata['feat']
            if self.gnn_type == "gat":
                for conv in self.convs[:-1]:
                    h = torch.mean(conv(g, h), dim=1)
                    h = F.relu(h)
                h = torch.mean(self.convs[-1](g, h), dim=1)
            else:
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


def run_experiment(num_epochs=100, lr=0.01, hidden_dim=256, num_layers=4, gnn_type="gcn"):
    g, labels, split_idx = load_dataset()
    evaluator = Evaluator(name='ogbn-arxiv')
    loss_func = nn.CrossEntropyLoss()
    model = GNN(128, hidden_dim, 40, num_layers, gnn_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    test_curve = []
    print("Started training ", gnn_type, "...", "\n", sep="")
    for epoch in range(num_epochs):
        loss = train(model, g, labels, split_idx, optimizer, loss_func)
        train_perf, valid_perf, test_perf = eval(model, g, labels, split_idx, evaluator)

        if (epoch + 1) % 10 == 0:
            print("===== {} epoch {} =====".format(gnn_type, epoch + 1))
            print('Loss:', loss)
            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
            print()

        test_curve.append(test_perf['acc'])
    print("Finished training", gnn_type, ".\n")

    return test_curve


def plot_results(gcn_curve, sage_curve, gat_curve):
    plt.plot(gcn_curve, label="GCN")
    plt.plot(sage_curve, label="SAGE")
    plt.plot(gat_curve, label="GAT")
    plt.title("Accuracy on the test dataset")
    plt.xlabel("Num. epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    num_epochs = 200
    gcn_curve = run_experiment(gnn_type="gcn", num_epochs=num_epochs)
    sage_curve = run_experiment(gnn_type="sage", num_epochs=num_epochs)
    gat_curve = run_experiment(gnn_type="gat", num_epochs=num_epochs)
    plot_results(gcn_curve, sage_curve, gat_curve)