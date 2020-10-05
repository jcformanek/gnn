import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.algorithms.simple_paths import shortest_simple_paths


class RoutingEnv():
    def __init__(self, nx_graph, num_routs=4, max_cap=200):
        self.nx_graph = nx_graph
        self.num_routs = num_routs
        self.max_capacity = max_cap
        self.capacity_matrix = None

    def _get_routs(self):
        src, dst = np.random.choice(self.nx_graph.number_of_nodes(), size=2, replace=False)
        all_paths = shortest_simple_paths(self.nx_graph, src, dst)
        rout_list = list(all_paths)[:self.num_routs] # num_routs shortest paths
        
        load = np.random.choice([8, 32, 64])

        rout_stack = []
        for rout in rout_list:
            rout_matrix = np.zeros_like(self.capacity_matrix)
            for i in range(len(rout) - 1):
                rout_matrix[rout[i], rout[i+1]] = load # allocate load
            rout_stack.append(rout_matrix)

        return rout_stack, rout_list, load


    def reset(self):
        self.capacity_matrix = adjacency_matrix(self.nx_graph).toarray() * self.max_capacity
        self.rout_stack, self.rout_list, self.load = self._get_routs()
        self.done = False
        obs = [self.capacity_matrix] + self.rout_stack
        obs = np.stack(obs)

        return obs

    
    def step(self, act):
        """
        act: integer in the interval [0, num_routs]
        """
        if self.done:
            raise ValueError("Env not initialised. Please reset.")
        
        elif act not in range(self.num_routs):
            raise ValueError("Not a valid action.")
            

        self.capacity_matrix = self.capacity_matrix - self.rout_stack[act]

        # check for overload
        rout = self.rout_list[act]
        for i in range(len(rout) - 1):
            if self.capacity_matrix[rout[i], rout[i+1]] < 0:
                self.done = True
                return None, 0, self.done

        # if no overload then return reward and next observation
        reward = self.load
        self.rout_stack, self.rout_list, self.load = self._get_routs()
        obs = [self.capacity_matrix] + self.rout_stack

        return obs, reward, self.done

    def sample_action(self):
        return np.random.choice(self.num_routs)

if __name__ == "__main__":
    g = nx.petersen_graph()
    env = RoutingEnv(g)

    returns = []
    for i in tqdm(range(1000)):
        total = 0
        env.reset()
        done = False
        while not done:
            obs, reward, done = env.step(env.sample_action())
            total += reward
        returns.append(total)

    plt.plot(returns)
    plt.show()
