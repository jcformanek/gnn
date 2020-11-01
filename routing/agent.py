import random
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import dgl

from dgqn import DGQN
from routing_env import RoutingEnv


class ReplayBuffer():

    def __init__(self, size=10000):
        self.obs_mem = []
        self.act_mem = []
        self.rew_mem = []
        self.next_obs_mem = []
        self.done_mem = []
        self.ctr = 0
        self.size = size

    def store(self, obs, act, rew, next_obs, done):
        if self.ctr < self.size:
            self.obs_mem.append(obs)
            self.act_mem.append(act)
            self.rew_mem.append(rew)
            self.next_obs_mem.append(next_obs)
            self.done_mem.append(done)
        else:
            idx = self.ctr % self.size
            self.obs_mem[idx] = obs
            self.act_mem[idx] = act
            self.rew_mem[idx] = rew
            self.next_obs_mem[idx] = next_obs
            self.done_mem[idx] = done

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
        done_sample = [self.done_mem[i] for i in idxs]

        return obs_sample, act_sample, rew_sample, next_obs_sample, done_sample



class DGQNAgent():
    def __init__(self, obs_dim, act_dim, dgl_graph, emb_dim=100, num_layers=3, lr=1e-3, gamma=0.99, memory_size=1000000,
            batch_size=64, epsilon=1, epsilon_min=0.01, epsilon_dec=5e-5, target_update_frequency=64):
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.Q_eval = DGQN(obs_dim, act_dim, emb_dim, num_layers)
        self.Q_target = DGQN(obs_dim, act_dim, emb_dim, num_layers)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.target_update_frequency = target_update_frequency
        self.learn_counter = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.optimizer = torch.optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.set_dgl_graph(dgl_graph)
        

    def set_dgl_graph(self, dgl_graph):
        self.dgl_graph_obs_list = []
        for i in range(self.batch_size):
            self.dgl_graph_obs_list.append(copy.deepcopy(dgl_graph))

        self.dgl_graph_next_obs_list = []
        for i in range(self.batch_size):
            self.dgl_graph_next_obs_list.append(copy.deepcopy(dgl_graph))

        self.dgl_graph = dgl_graph

    
    def update_target(self):
        if self.learn_counter % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())


    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec


    def store_transition(self, obs, act, rew, next_obs, done):
        self.memory.store(obs, act, rew, next_obs, done)

    
    def choose_action(self, obs):
        if np.random.sample() < self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            self.dgl_graph.edata['obs'] = obs
            return torch.argmax(self.Q_eval(self.dgl_graph)).item()


    def choose_best_action(self, obs):
        self.dgl_graph.edata['obs'] = obs
        return torch.argmax(self.Q_eval(self.dgl_graph)).item()


    def sample_memory_batch(self):
        obs_list, act_list, rew_list, next_obs_list, done_list = self.memory.sample(self.batch_size)
        
        for i, obs in enumerate(obs_list):
            self.dgl_graph_obs_list[i].edata['obs'] = obs
        obs_batch = dgl.batch(self.dgl_graph_obs_list)

        for i, obs in enumerate(next_obs_list):
            self.dgl_graph_next_obs_list[i].edata['obs'] = obs
        next_obs_batch = dgl.batch(self.dgl_graph_next_obs_list)

        act_batch = torch.tensor(act_list, dtype=torch.long)
        rew_batch = torch.tensor(rew_list, dtype=torch.float)
        done_batch = torch.tensor(done_list, dtype=torch.int16)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch

    
    def learn(self):
        if self.memory.ctr < self.batch_size:
            return

        self.optimizer.zero_grad()
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.sample_memory_batch()
        idxs = np.arange(len(act_batch))
        q_pred = self.Q_eval(obs_batch)[idxs, act_batch]
        q_next = self.Q_target(next_obs_batch).max(dim=1)[0]
        q_target = rew_batch + (1 - done_batch) * self.gamma * q_next
        loss = self.loss_fn(q_target, q_pred)
        loss.backward()
        self.optimizer.step()
        self.update_target()
        self.decrement_epsilon()

def train(agent, env, num_epochs=100):
    scores = []
    avg_scores = []
    for e in tqdm(range(num_epochs)):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            act = agent.choose_action(obs)
            next_obs, rew, done = env.step(act)
            agent.store_transition(obs, act, rew, next_obs, done)
            agent.learn()
            obs = next_obs
            score += rew
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        if e % 100 == 0:
            print()
            print('epoch: ', e, ' average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    print('done')
    
    return avg_scores

@torch.no_grad()
def evaluate(agent, env, num_rounds=300):
    scores = []
    for e in tqdm(range(num_rounds)):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            act = agent.choose_best_action(obs)
            next_obs, rew, done = env.step(act)
            score += rew
        scores.append(score)
    
    avg_score = np.mean(scores)
    return avg_score

def evaluate_random_agent(env, num_rounds=300):
    scores = []
    for e in tqdm(range(num_rounds)):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            act = env.sample_action()
            next_obs, rew, done = env.step(act)
            score += rew
        scores.append(score)
    
    avg_score = np.mean(scores)
    return avg_score

if __name__ == '__main__':
    g = nx.petersen_graph()
    env = RoutingEnv(g)
    dgl_graph = env.get_dgl_graph()
    agent = DGQNAgent(env.obs_dim, env.act_dim, dgl_graph)
    train(agent, env)