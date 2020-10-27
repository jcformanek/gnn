import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
import dgl.function as fn

import ogb
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


### Load the molecule dataset
def load_dataset():
    dataset = DglGraphPropPredDataset(name = 'ogbg-molhiv')
    split_idx = dataset.get_idx_split()

    ### Collate function for dataloaders
    def _collate_fn(batch):
        # batch is a list of tuple (graph, label)
        graphs = [e[0] for e in batch]
        g = dgl.batch(graphs)
        labels = [e[1] for e in batch]
        labels = torch.stack(labels, 0)
        return g, labels

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=_collate_fn)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)

    ### Return dataloaders
    return train_loader, valid_loader, test_loader


class GnnConv(nn.Module):
    def __init__(self, emb_dim, agg_type):
        super(GnnConv, self).__init__()
        """
        emb_dim: int
        agg_type: string
        """

        ### Edge feature encoder
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

        ### Neighbourhood Message Function
        self.gnn_msg = fn.v_mul_e(lhs_field='h', rhs_field='he', out='m')

        ### Neighbourhood Aggregator Function
        if agg_type == 'sum':
            self.gnn_agg = fn.sum(msg='m', out='h')
        elif agg_type == 'max':
            self.gnn_agg = fn.max(msg='m', out='h')
        elif agg_type == 'mean':
            self.gnn_agg = fn.mean(msg='m', out='h')
        else:
            ValueError('Undefined neighbourhood aggregator type called {}'.format(agg_type))

        ### State transformer
        self.linear = nn.Linear(emb_dim, emb_dim)


    def forward(self, g, h):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = h
            xe = g.edata['feat']

            ### Transformation of edge feature vector
            g.edata['he'] = self.bond_encoder(xe)

            ### Neighbourhood aggregation
            g.update_all(self.gnn_msg, self.gnn_agg)

            h = g.ndata['h']

            ### Transformation of state vector
            h = self.linear(h)

            return h


class Gnn(nn.Module):
    def __init__(self, emb_dim, num_layers, agg_type):
        super(Gnn, self).__init__()
        """
        emb_dim: int
        num_layers: int
        agg_type: string
        """

        ### Node feature vector encoder
        self.atom_encoder = AtomEncoder(emb_dim=emb_dim)

        ### List of graph conv layers
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GnnConv(emb_dim, agg_type))

        ### Graph classifier layer
        self.classifier = nn.Linear(emb_dim, 1)
        

    def forward(self, g):
        with g.local_scope():
            x = g.ndata['feat']

            ### Transformation of node feature vector
            h = self.atom_encoder(x)

            ### Message passing among graph nodes
            for layer in range(self.num_layers):
                h = self.convs[layer](g, h)
                h = F.relu(h)

            ### Graph Readout
            g.ndata['h'] = h
            hg = dgl.sum_nodes(g, 'h')
            y = self.classifier(hg)
            return y


### Train the model for one epoch
def train(model, dataloader, optimizer, loss_func):
    model.train()
    for batch_graph, labels in tqdm(dataloader):
        pred = model(batch_graph)
        optimizer.zero_grad()
        pred = pred.view(-1)
        labels = labels.view(-1)
        loss = loss_func(pred.to(torch.float32), labels.to(torch.float32))
        loss.backward()
        optimizer.step()


### Evaluate the model on a dataset
def eval(model, dataloader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for batch_graph, labels in tqdm(dataloader):

        with torch.no_grad():
            pred = model(batch_graph)

        y_true.append(labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    ### OGB evaluator
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    output_dict = evaluator.eval(input_dict)

    return output_dict


def run_experiment(num_epochs=10, lr=0.001, emb_dim=100, num_layers=3, agg_type='sum'):
    evaluator = Evaluator("ogbg-molhiv")

    train_loader, valid_loader, test_loader = load_dataset()

    model = Gnn(emb_dim=emb_dim, num_layers=num_layers, agg_type=agg_type)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.BCEWithLogitsLoss()

    test_curve = []

    for epoch in range(num_epochs):
        print("===== {} epoch {} =====".format(agg_type, epoch + 1))
        print('Training...')
        train(model, train_loader, optimizer, loss_func)

        print('Evaluating...')
        train_perf = eval(model, train_loader, evaluator)
        valid_perf = eval(model, valid_loader, evaluator)
        test_perf = eval(model, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        test_curve.append(test_perf['rocauc'])

    return test_curve


def plot_results(sum_curve, mean_curve, max_curve):
    plt.plot(sum_curve, label="Sum")
    plt.plot(mean_curve, label="Mean")
    plt.plot(max_curve, label="Max")
    plt.title("Accuracy on the test dataset")
    plt.xlabel("Num. epochs")
    plt.ylabel("ROC-ACC")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    num_epochs = 20
    sum_curve = run_experiment(num_epochs=num_epochs, agg_type='sum')
    mean_curve = run_experiment(num_epochs=num_epochs, agg_type='mean')
    max_curve = run_experiment(num_epochs=num_epochs, agg_type='max')
    plot_results(sum_curve, mean_curve, max_curve)