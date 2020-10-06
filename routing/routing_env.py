import numpy as np
import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.simple_paths import shortest_simple_paths
from networkx.linalg.graphmatrix import adjacency_matrix
import copy

class RoutingEnv():
    def __init__(self, nx_graph, num_routes=4, max_capacity=200):
        self.max_capacity = max_capacity
        self.num_routes = num_routes
        self.nx_graph = nx_graph
        self.dgl_graph = dgl.from_networkx(self.nx_graph)
        self.n = self.dgl_graph.number_of_nodes()
        self.m = self.dgl_graph.number_of_edges()
        self.load = None
        self.routes = None
        self.done = True

    @torch.no_grad()
    def _get_next_load(self):
        self.load = np.random.choice([8, 32, 64])
        src, dst = np.random.choice(self.n, size=2, replace=False)
        self.routes  = list(shortest_simple_paths(self.nx_graph, src, dst))[:self.num_routes]
        self.dgl_graph.edata["feat"][:, 1:] = torch.zeros_like(self.dgl_graph.edata["feat"][:, 1:])
        for i, route in enumerate(self.routes):
            src = route[:-1]
            dst = route[1:]
            self.dgl_graph.edata["feat"][self.dgl_graph.edge_ids(src, dst), i+1] = torch.zeros(len(src)) + self.load

    @torch.no_grad()
    def reset(self):
        """ 
        nx_G: a Networkx graph giving the topology of the network to train on
        """
        self.dgl_graph.edata["feat"] = torch.zeros((self.m, self.num_routes + 1))
        self.dgl_graph.edata["feat"][:, 0] = torch.zeros(self.m) + self.max_capacity
        self._get_next_load()
        self.done = False
        obs = self.dgl_graph.edata["feat"].detach().clone()
        return obs, self.dgl_graph

    @torch.no_grad()
    def step(self, act):

        if self.done or act not in range(self.num_routes):
            raise ValueError("Please reset environment")

        route = self.routes[act]
        src = route[:-1]
        dst = route[1:]
        edge_idx = self.dgl_graph.edge_ids(src, dst)
        self.dgl_graph.edata["feat"][edge_idx, 0] -= self.load
        for val in self.dgl_graph.edata["feat"][edge_idx, 0]:
            if val < 0:
                self.done = True
                obs = self.dgl_graph.edata["feat"].detach().clone()
                return obs, 0, self.done # obs, rew, done
        reward = self.load
        self._get_next_load()
        obs = self.dgl_graph.edata["feat"].detach().clone()
        return obs, reward, self.done

    def sample_action(self):
        return np.random.choice(self.num_routes)    


if __name__ == '__main__':
    P = nx.petersen_graph()
    #nx.draw(P, with_labels=True)
    env = RoutingEnv(P)
    G = env.reset()
    returns = []
    for i in range(200):
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
