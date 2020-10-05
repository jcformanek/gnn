import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.simple_paths import shortest_simple_paths
from networkx.linalg.graphmatrix import adjacency_matrix

class RoutingEnv():
    def __init__(self, nx_g, k=4, max_b=200):
        self.k = k
        self.max_bandwidth = max_b
        self.nx_g = nx_g
        self.adjacency_matrix = adjacency_matrix(self.nx_g)
        self.n = self.nx_g.number_of_nodes()
        self.traffic_matrix = None
        self.actions = None
        self.shortest_paths = None
        self.bandwidth = None
        self.done = True


    def get_actions(self):
        src, dst = np.random.choice(self.G.number_of_nodes(), size=2, replace=False)
        all_paths = shortest_simple_paths(self.G, src, dst)
        shortest_paths = []
        for i, p in enumerate(all_paths):
            if i >= self.k:
                break
            shortest_paths.append(p)
        
        bandwidth = np.random.choice([8, 32, 64])
        actions = []
        for P in shortest_paths:
            actions.append(np.zeros_like(self.traffic_matrix))
            action = actions[-1]
            for i in range(len(P) - 1):
                action[P[i], P[i+1]] = bandwidth # this must be done in a better way. 
                action[P[i+1], P[i]] = bandwidth # capacity is shared in both directions

        return actions, shortest_paths, bandwidth


    def reset(self):
        """ 
        nx_G: a Networkx graph giving the topology of the network to train on
        """
        self.traffic_matrix = np.zeros((self.n, self.n), dtype=np.int16)
        self.actions, self.shortest_paths, self.bandwidth = self.get_actions()
        self.done = False
        observation = self.actions + [self.traffic_matrix] + [self.adjacency_matrix]

        return observation

    
    def step(self, act):
        """
        act: int in the interval [0,k]
        """
        if self.done or act not in range(self.k+1):
            print("Error")
            return

        self.traffic_matrix = self.traffic_matrix + self.actions[act]
        P = self.shortest_paths[act]
        for i in range(len(P) - 1):
            if self.traffic_matrix[P[i], P[i+1]] > self.max_bandwidth:
                self.done = True
                return [], 0, self.done

        reward = self.bandwidth
        self.actions, self.shortest_paths, self.bandwidth = self.get_actions()
        observation = self.actions + [self.traffic_matrix] + [self.adjacency_matrix]
        return observation, reward, self.done

    def sample_action(self):
        return np.random.choice(self.k)      

if __name__ == '__main__':
    P = nx.petersen_graph()
    #nx.draw(P, with_labels=True)
    env = RoutingEnv(P)
    returns = []
    for i in range(1000):
        total = 0
        env.reset()
        done = False
        while not done:
            obs, reward, done = env.step(env.sample_action())
            total += reward
        returns.append(total)

    plt.plot(returns)
    plt.show()
