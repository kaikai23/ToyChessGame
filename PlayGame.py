import numpy as np
from Chess_env import *
import torch.nn as nn
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path

# env=Chess_Env(N_grid=4)
# S,X,allowed_a=env.Initialise_game()
# N_a=np.shape(allowed_a)[0]   # TOTAL NUMBER OF POSSIBLE ACTIONS
# N_in=np.shape(X)[0]    ## INPUT SIZE
# N_h=200                ## NUMBER OF HIDDEN NODES


## DEFINE YOUR NEURAL NETWORK...
class Q_Net(nn.Module):
    def __init__(self,in_dim, hidden_dim, out_dim, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 activation,
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x, mask):
        return self.net(x) * mask

# HYPERPARAMETERS SUGGESTED (FOR A GRID SIZE OF 4)

# epsilon_0 = 0.2     # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
# beta = 0.00005      # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
# gamma = 0.85        # THE DISCOUNT FACTOR
# eta = 0.0035        # THE LEARNING RATE
#
# # N_episodes = 100000 # THE NUMBER OF GAMES TO BE PLAYED
# N_episodes = 100000
#
# # SAVING VARIABLES
# R_save = np.zeros([N_episodes, 1])
# N_moves_save = np.zeros([N_episodes, 1])

# DEFINE THE EPSILON-GREEDY POLICY
def EpsilonGreedy_Policy(Qvalues, allowed_a, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.permutation(np.where(allowed_a)[0])[0]
    else:
        assert type(Qvalues) is torch.Tensor
        Qvalues = Qvalues.detach().numpy()
        masked_Qvalues = np.ma.masked_equal(Qvalues, 0.0, copy=False)
        a = masked_Qvalues.argmax()
    return a

class Agent:
    def __init__(self, env):
        self.env = env
        self.Q_Net = Q_Net(in_dim=58, hidden_dim=200, out_dim=32)
        self.eta = 0.0035
        self.gamma = 0.85
        self.beta = 0.00005
        self.epsilon_0 = 0.2
        self.N_episodes = 100000
        self.R_save = np.zeros([self.N_episodes, 1])
        self.N_moves_save = np.zeros([self.N_episodes, 1])
        self.optimizer = torch.optim.SGD(self.Q_Net.parameters(), lr=self.eta)
        self.optimizer_name = 'SGD'

    def reset_Network_Parameters(self):
        self.Q_Net = Q_Net(in_dim=58, hidden_dim=200, out_dim=32)
        if self.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.Q_Net.parameters(), lr=self.eta)
        elif self.optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.Q_Net.parameters(), lr=self.eta)
        # MUST BE FO

    def reset_R_and_N_moves(self):
        self.R_save = np.zeros([self.N_episodes, 1])
        self.N_moves_save = np.zeros([self.N_episodes, 1])

    def set_N_episodes(self, N):
        self.N_episodes = N

    def set_discount_factor(self, gamma):
        self.gamma = gamma

    def set_decay_speed(self, beta):
        self.beta = beta

    def set_optimizer(self, lr, name='SGD'):
        self.eta = lr
        self.optimizer_name = name
        if name == 'SGD':
            self.optimizer = torch.optim.SGD(self.Q_Net.parameters(), lr=lr)
        elif name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.Q_Net.parameters(), lr=lr)
        else:
            raise 'ERROR: Method Not Implemented'

    def reset_all(self):
        self.Q_Net = Q_Net(in_dim=58, hidden_dim=200, out_dim=32)
        self.optimizer = torch.optim.SGD(self.Q_Net.parameters(), lr=self.eta)
        self.eta = 0.0035
        self.gamma = 0.85
        self.beta = 0.00005
        self.epsilon_0 = 0.2
        self.N_episodes = 100000
        self.R_save = np.zeros([self.N_episodes, 1])
        self.N_moves_save = np.zeros([self.N_episodes, 1])

    def train_q_learning(self):
        self.reset_R_and_N_moves()
        self.reset_Network_Parameters()
        print("=========== Starting to train ============ ")
        print("----------- Method: Q Learning ----------- ")
        print(f'number of episodes: {self.N_episodes}\ndicount factor: {self.gamma}\ndecay factor: {self.beta}')
        for n in tqdm(range(self.N_episodes)):
            epsilon_f = self.epsilon_0 / (1 + self.beta * n)
            Done = 0
            i = 1
            S, X, allowed_a = self.env.Initialise_game()
            while Done == 0:  ## START THE EPISODE
                Q_values = self.Q_Net(torch.from_numpy(X).float(), torch.from_numpy(allowed_a).squeeze())
                a = EpsilonGreedy_Policy(Q_values, allowed_a, epsilon_f)
                S_next, X_next, allowed_a_next, R, Done = self.env.OneStep(a)
                if Done == 1:
                    self.R_save[n] = np.copy(R)
                    self.N_moves_save[n] = np.copy(i)
                    self.optimizer.zero_grad()
                    loss = 0.5 * (Q_values[a] - R) ** 2
                    loss.backward()
                    self.optimizer.step()
                    break
                else:
                    Q_values_next = self.Q_Net(torch.from_numpy(X_next).float(), torch.from_numpy(allowed_a_next).squeeze())
                    self.optimizer.zero_grad()
                    loss = 0.5 * (Q_values[a] - R - self.gamma * Q_values_next.max()) ** 2
                    loss.backward()
                    self.optimizer.step()
                X = np.copy(X_next)
                allowed_a = np.copy(allowed_a_next)
                i += 1
        print('=========== Finished training ============\n ')
        return self.R_save, self.N_moves_save

    def train_sarsa(self):
        self.reset_R_and_N_moves()
        self.reset_Network_Parameters()
        print("=========== Starting to train ============ ")
        print("----------- Method: SARSA ----------- ")
        print(f'number of episodes: {self.N_episodes}\ndicount factor: {self.gamma}\ndecay factor: {self.beta}')
        for n in tqdm(range(self.N_episodes)):
            epsilon_f = self.epsilon_0 / (1 + self.beta * n)
            Done = 0
            i = 1
            S, X, allowed_a = self.env.Initialise_game()
            Q_values = self.Q_Net(torch.from_numpy(X).float(), torch.from_numpy(allowed_a).squeeze())
            a = EpsilonGreedy_Policy(Q_values, allowed_a, epsilon_f)
            while Done == 0:
                Q_values = self.Q_Net(torch.from_numpy(X).float(), torch.from_numpy(allowed_a).squeeze())
                S_next, X_next, allowed_a_next, R, Done = self.env.OneStep(a)
                if Done ==1:
                    self.R_save[n] = np.copy(R)
                    self.N_moves_save[n] = np.copy(i)
                    self.optimizer.zero_grad()
                    loss = 0.5 * (Q_values[a] - R) ** 2
                    loss.backward()
                    self.optimizer.step()
                    break
                Q_values_next = self.Q_Net(torch.from_numpy(X_next).float(), torch.from_numpy(allowed_a_next).squeeze())
                a_next = EpsilonGreedy_Policy(Q_values_next, allowed_a_next, epsilon_f)
                self.optimizer.zero_grad()
                loss = 0.5 * (Q_values[a] - R - self.gamma * Q_values_next[a_next]) ** 2
                loss.backward()
                self.optimizer.step()
                X = np.copy(X_next)
                allowed_a = np.copy(allowed_a_next)
                a = a_next
                i += 1
        print('=========== Finished training ============\n ')
        return self.R_save, self.N_moves_save


env = Chess_Env(N_grid=4)
agent = Agent(env)
agent.reset_all()

# Ensure the results folder exists, otherwise create one.
Path('./results').mkdir(parents=True, exist_ok=True)

# Structure to save results
Results = dict()
Results['Q_learning'] = dict()
Results['SARSA'] = dict()

# Q learning, default hyperparameters
Results['Q_learning']['default'] = dict()
Results['Q_learning']['default']['R'], Results['Q_learning']['default']['N_moves'] = agent.train_q_learning()
np.save('results/R_q_default.npy', Results['Q_learning']['default']['R'])
np.save('results/N_q_default.npy', Results['Q_learning']['default']['N_moves'])

# SARSA, default hyperparameters
Results['SARSA']['default'] = dict()
Results['SARSA']['default']['R'], Results['SARSA']['default']['N_moves'] = agent.train_sarsa()
np.save('results/R_sarsa_default.npy', Results['SARSA']['default']['R'])
np.save('results/N_sarsa_default.npy', Results['SARSA']['default']['N_moves'])

# DISCOUNT FACTOR: gamma = 0.75
agent.set_discount_factor(0.75)
Results['Q_learning']['gamma=0.75'] = dict()
Results['Q_learning']['gamma=0.75']['R'], Results['Q_learning']['gamma=0.75']['N_moves'] = agent.train_q_learning()
np.save('results/R_q_gamma0.75.npy', Results['Q_learning']['gamma=0.75']['R'])
np.save('results/N_q_gamma0.75.npy', Results['Q_learning']['gamma=0.75']['N_moves'])

Results['SARSA']['gamma=0.75'] = dict()
Results['SARSA']['gamma=0.75']['R'], Results['SARSA']['gamma=0.75']['N_moves'] = agent.train_sarsa()
np.save('results/R_sarsa_gamma0.75.npy', Results['SARSA']['gamma=0.75']['R'])
np.save('results/N_sarsa_gamma0.75.npy', Results['SARSA']['gamma=0.75']['N_moves'])

#  DISCOUNT FACTOR: gamma = 0.95
agent.set_discount_factor(0.95)
Results['Q_learning']['gamma=0.95'] = dict()
Results['Q_learning']['gamma=0.95']['R'], Results['Q_learning']['gamma=0.95']['N_moves'] = agent.train_q_learning()
np.save('results/R_q_gamma0.95.npy', Results['Q_learning']['gamma=0.95']['R'])
np.save('results/N_q_gamma0.95.npy', Results['Q_learning']['gamma=0.95']['N_moves'])

Results['SARSA']['gamma=0.95'] = dict()
Results['SARSA']['gamma=0.95']['R'], Results['SARSA']['gamma=0.95']['N_moves'] = agent.train_sarsa()
np.save('results/R_sarsa_gamma0.95.npy', Results['SARSA']['gamma=0.95']['R'])
np.save('results/N_sarsa_gamma0.95.npy', Results['SARSA']['gamma=0.95']['N_moves'])

# DECAY FACTOR: beta = 0.0005
agent.reset_all()
agent.set_decay_speed(0.0005)
Results['Q_learning']['beta=0.0005'] = dict()
Results['Q_learning']['beta=0.0005']['R'], Results['Q_learning']['beta=0.0005']['N_moves'] = agent.train_q_learning()
np.save('results/R_q_beta0.0005.npy', Results['Q_learning']['beta=0.0005']['R'])
np.save('results/N_q_beta0.0005.npy', Results['Q_learning']['beta=0.0005']['N_moves'])

Results['SARSA']['beta=0.0005'] = dict()
Results['SARSA']['beta=0.0005']['R'], Results['SARSA']['beta=0.0005']['N_moves'] = agent.train_sarsa()
np.save('results/R_sarsa_beta0.0005.npy', Results['SARSA']['beta=0.0005']['R'])
np.save('results/N_sarsa_beta0.0005.npy', Results['SARSA']['beta=0.0005']['N_moves'])

# DECAY FACTOR: beta = 0.000005
agent.reset_all()
agent.set_decay_speed(0.000005)
Results['Q_learning']['beta=0.000005'] = dict()
Results['Q_learning']['beta=0.000005']['R'], Results['Q_learning']['beta=0.000005']['N_moves'] = agent.train_q_learning()
np.save('results/R_q_beta0.000005.npy', Results['Q_learning']['beta=0.000005']['R'])
np.save('results/N_q_beta0.000005.npy', Results['Q_learning']['beta=0.000005']['N_moves'])

Results['SARSA']['beta=0.000005'] = dict()
Results['SARSA']['beta=0.000005']['R'], Results['SARSA']['beta=0.000005']['N_moves'] = agent.train_sarsa()
np.save('results/R_sarsa_beta0.000005.npy', Results['SARSA']['beta=0.000005']['R'])
np.save('results/N_sarsa_beta0.000005.npy', Results['SARSA']['beta=0.000005']['N_moves'])

# # TRAINING LOOP BONE STRUCTURE...
# Q_Net = Q_Net(N_in, N_h, N_a)
# optimizer = torch.optim.SGD(Q_Net.parameters(), lr=eta)
# # I WROTE FOR YOU A RANDOM AGENT (THE RANDOM AGENT WILL BE SLOWER TO GIVE CHECKMATE THAN AN OPTIMISED ONE,
# # SO DON'T GET CONCERNED BY THE TIME IT TAKES), CHANGE WITH YOURS ...
#
# for n in tqdm(range(N_episodes)):
#     epsilon_f = epsilon_0 / (1 + beta * n)  ## DECAYING EPSILON
#     Done = 0  ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
#     i = 1  ## COUNTER FOR NUMBER OF ACTIONS
#     S, X, allowed_a = env.Initialise_game()  ## INITIALISE GAME
#     while Done == 0:  ## START THE EPISODE
#         ## THIS IS A RANDOM AGENT, CHANGE IT...
#         Q_values = Q_Net(torch.from_numpy(X).float(), torch.from_numpy(allowed_a).squeeze())
#         a = EpsilonGreedy_Policy(Q_values, allowed_a, epsilon_f)
#         S_next, X_next, allowed_a_next, R, Done = env.OneStep(a)
#
#         ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
#         if Done == 1:
#             R_save[n] = np.copy(R)
#             N_moves_save[n] = np.copy(i)
#
#             optimizer.zero_grad()
#             loss = 0.5 * (Q_values[a] - R)**2
#             loss.backward()
#             optimizer.step()
#             break
#
#         # IF THE EPISODE IS NOT OVER...
#         else:
#             Q_values_next = Q_Net(torch.from_numpy(X_next).float(), torch.from_numpy(allowed_a_next).squeeze())
#             optimizer.zero_grad()
#             loss = 0.5 * (Q_values[a] - R - gamma * Q_values_next.max()) ** 2
#             loss.backward()
#             optimizer.step()
#
#
#         # NEXT STATE AND CO. BECOME ACTUAL STATE...
#         S = np.copy(S_next)
#         X = np.copy(X_next)
#         allowed_a = np.copy(allowed_a_next)
#
#         i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

# PLOT THE TOTAL REWARD PER EPISODE AS TRAINING PROGRESSES
def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# plt.plot(R_save)
# plt.show()
# plt.plot(moving_average(R_save, n=1000))
# plt.show()
# plt.plot(moving_average(N_moves_save, n=1000))
# plt.show()
# print('Random_Agent, Average reward:',np.mean(R_save),'Number of steps: ',np.mean(N_moves_save))

# PLOT DEFAULT Q LEARNING AND SARSA
plt.figure()
plt.plot(moving_average(Results['Q_learning']['default']['R']), label='Q Learning')
plt.plot(moving_average(Results['SARSA']['default']['R']), label='SARSA')
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.legend()
plt.show()

qwerty=1
print('dont stop, I have programs to run')
print(f'qwerty={qwerty}')
