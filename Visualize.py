import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


results = './results'
results_folder = Path(results)
assert results_folder.is_dir()


# Q Learning vs. SARSA (both use default hyperparameters)
r1 = np.load('results/R_q_default.npy')
r2 = np.load('results/R_sarsa_default.npy')

plt.figure()
plt.plot(moving_average(r1, 1000), label='Q Learning')
plt.plot(moving_average(r2, 1000), label='SARSA')
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.title('Q Learning vs. SARSA )')
plt.legend()
plt.show()

n1 = np.load('results/N_q_default.npy')
n2 = np.load('results/N_sarsa_default.npy')

plt.figure()
plt.plot(moving_average(n1, 1000), label='Q Learning')
plt.plot(moving_average(n2, 1000), label='SARSA')
plt.xlabel('Episode Number')
plt.ylabel('Number of Moves')
plt.title('Q Learning vs. SARSA )')
plt.legend()
plt.show()


# Q Learning, Ablation: gamma 0.75 vs 0.85 vs 0.95
r1 = np.load('results/R_q_gamma0.75.npy')
r2 = np.load('results/R_q_default.npy')
r3 = np.load('results/R_q_gamma0.95.npy')

plt.figure()
plt.plot(moving_average(r1, 1000), label='gamma=0.75 ')
plt.plot(moving_average(r2, 1000), label='gamma=0.85 ')
plt.plot(moving_average(r3, 1000), label='gamma=0.95 ')
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.title('Discount Factor in Q Learning')
plt.legend()
plt.show()

n1 = np.load('results/N_q_gamma0.75.npy')
n2 = np.load('results/N_q_default.npy')
n3 = np.load('results/N_q_gamma0.95.npy')

plt.figure()
plt.plot(moving_average(n1, 1000), label='gamma=0.75')
plt.plot(moving_average(n2, 1000), label='gamma=0.85')
plt.plot(moving_average(n3, 1000), label='gamma=0.95')
plt.xlabel('Episode Number')
plt.ylabel('Number of Moves')
plt.title('Discount Factor in Q Learning')
plt.legend()
plt.show()


# SARSA, Ablation: gamma 0.75 vs 0.85 vs 0.95
r1 = np.load('results/R_sarsa_gamma0.75.npy')
r2 = np.load('results/R_sarsa_default.npy')
r3 = np.load('results/R_sarsa_gamma0.95.npy')

plt.figure()
plt.plot(moving_average(r1, 1000), label='gamma=0.75 ')
plt.plot(moving_average(r2, 1000), label='gamma=0.85 ')
plt.plot(moving_average(r3, 1000), label='gamma=0.95 ')
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.title('Discount Factor in SARSA')
plt.legend()
plt.show()

n1 = np.load('results/N_sarsa_gamma0.75.npy')
n2 = np.load('results/N_sarsa_default.npy')
n3 = np.load('results/N_sarsa_gamma0.95.npy')

plt.figure()
plt.plot(moving_average(n1, 1000), label='gamma=0.75')
plt.plot(moving_average(n2, 1000), label='gamma=0.85')
plt.plot(moving_average(n3, 1000), label='gamma=0.95')
plt.xlabel('Episode Number')
plt.ylabel('Number of Moves')
plt.title('Discount Factor in SARSA')
plt.legend()
plt.show()

# Q Learning, Ablation: beta 5e-4 vs 5e-5 vs 5e-6
r1 = np.load('results/R_q_beta0.0005.npy')
r2 = np.load('results/R_q_default.npy')
r3 = np.load('results/R_q_beta0.000005.npy')

plt.figure()
plt.plot(moving_average(r1, 1000), label='beta=5e-4 ')
plt.plot(moving_average(r2, 1000), label='beta=5e-5 ')
plt.plot(moving_average(r3, 1000), label='beta=5e-6 ')
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.title('Epsilon Decay in Q Learning')
plt.legend()
plt.show()

n1 = np.load('results/N_q_beta0.0005.npy')
n2 = np.load('results/N_q_default.npy')
n3 = np.load('results/N_q_beta0.000005.npy')

plt.figure()
plt.plot(moving_average(n1, 1000), label='beta=5e-4')
plt.plot(moving_average(n2, 1000), label='beta=5e-5')
plt.plot(moving_average(n3, 1000), label='beta=5e-6')
plt.xlabel('Episode Number')
plt.ylabel('Number of Moves')
plt.title('Epsilon Decay in Q Learning')
plt.legend()
plt.show()


# SARSA, Ablation: beta 5e-4 vs 5e-5 vs 5e-6
r1 = np.load('results/R_sarsa_beta0.0005.npy')
r2 = np.load('results/R_sarsa_default.npy')
r3 = np.load('results/R_sarsa_beta0.000005.npy')

plt.figure()
plt.plot(moving_average(r1, 1000), label='beta=5e-4 ')
plt.plot(moving_average(r2, 1000), label='beta=5e-5 ')
plt.plot(moving_average(r3, 1000), label='beta=5e-6 ')
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.title('Epsilon Decay in SARSA')
plt.legend()
plt.show()

n1 = np.load('results/N_sarsa_beta0.0005.npy')
n2 = np.load('results/N_sarsa_default.npy')
n3 = np.load('results/N_sarsa_beta0.000005.npy')

plt.figure()
plt.plot(moving_average(n1, 1000), label='beta=5e-4')
plt.plot(moving_average(n2, 1000), label='beta=5e-5')
plt.plot(moving_average(n3, 1000), label='beta=5e-6')
plt.xlabel('Episode Number')
plt.ylabel('Number of Moves')
plt.title('Epsilon Decay in SARSA')
plt.legend()
plt.show()

