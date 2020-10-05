import numpy as np
import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.simple_paths import shortest_simple_paths
from networkx.linalg.graphmatrix import adjacency_matrix
import copy

class GraphRoutingEnv():
    def __init__(self, nx_g, k=4, max_bw=200, feature_size=10):
        self.k = k
        self.max_bw = max_bw
        self.nx_g = nx_g
        self.dgl_g = dgl.DGLGraph(self.nx_g)
        self.n = self.dgl_g.number_of_nodes()
        self.m = self.dgl_g.number_of_edges()
        self.bw = None
        self.paths = None
        self.done = True
        self.feature_size = feature_size


    def alloc_demands(self):
        self.bw = np.random.choice([8, 32, 64])
        src, dst = np.random.choice(self.n, size=2, replace=False)
        paths  = list(shortest_simple_paths(self.nx_g, src, dst))[:self.k]
        self.paths = paths
        for i, path in enumerate(paths):
            src = path[:-1] + path[1:]
            dst = path[1:] + path[:-1]
            self.dgl_g.edata["feat"][self.dgl_g.edge_ids(src, dst), i+1] = torch.zeros(len(src))+self.bw


    def reset(self):
        """ 
        nx_G: a Networkx graph giving the topology of the network to train on
        """
        self.dgl_g.edata["feat"] = torch.zeros((self.m, self.feature_size))
        self.dgl_g.edata["feat"][:, 0] = torch.zeros(self.m) + self.max_bw
        self.alloc_demands()
        self.done = False

        return copy.deepcopy(self.dgl_g)


    def step(self, act):

        if self.done or act not in range(self.k):
            raise ValueError("Please reset environment")

        path = self.paths[act]
        src = path[:-1] + path[1:]
        dst = path[1:] + path[:-1]
        self.dgl_g.edata["feat"][self.dgl_g.edge_ids(src, dst), 0] -= self.bw
        for val in self.dgl_g.edata["feat"][self.dgl_g.edge_ids(src, dst), 0]:
            if val < 0:
                self.done = True
                return self.dgl_g, 0, True # obs, rew, done
        reward = self.bw
        self.alloc_demands()
        return copy.deepcopy(self.dgl_g), reward, False

    def sample_action(self):
        return np.random.choice(self.k)    


if __name__ == '__main__':
    P = nx.petersen_graph()
    #nx.draw(P, with_labels=True)
    env = GraphRoutingEnv(P)
    returns = []
    for i in range(1000):
        total = 0
        if i % 100 == 0:
            print(i)
        env.reset()
        done = False
        while not done:
            obs, reward, done = env.step(env.sample_action())
            total += reward
        returns.append(total)

    plt.plot(returns)
    plt.show()
