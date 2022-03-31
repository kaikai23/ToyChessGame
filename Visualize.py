import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def visualize():
    results = './results'
    results_folder = Path(results)
    assert results_folder.is_dir()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Q Learning vs. SARSA')
    # ax1.plot(moving_average(r1, 1000), label='Q Learning')
    # ax1.plot(moving_average(r2, 1000), label='SARSA')
    # ax1.set(xlabel='Episode Number', ylabel='Reward')
    # ax2.plot(moving_average(n1, 1000), label='Q Learning')
    # ax2.plot(moving_average(n2, 1000), label='SARSA')
    # ax2.set(xlabel='Episode Number', ylabel='Number of Moves')
    # plt.show()


    # Q Learning vs. SARSA (both use default hyperparameters) + vs. new state representation
    r1 = np.load('results/R_q_default.npy')
    r2 = np.load('results/R_sarsa_default.npy')
    r3 = np.load('results/R_q_new.npy')
    r4 = np.load('results/R_sarsa_new.npy')

    plt.figure()
    plt.plot(moving_average(r1, 2000), label='Q Learning')
    plt.plot(moving_average(r2, 2000), label='SARSA')
    plt.plot(moving_average(r3, 2000), label='Q Learning + new state')
    plt.plot(moving_average(r4, 2000), label='SARSA + new state')
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.ylim(0.8, 1)
    plt.title('New state representation')
    plt.legend()
    plt.show()

    n1 = np.load('results/N_q_default.npy')
    n2 = np.load('results/N_sarsa_default.npy')
    n3 = np.load('results/N_q_new.npy')
    n4 = np.load('results/N_sarsa_new.npy')

    plt.figure()
    plt.plot(moving_average(n1, 2000), label='Q Learning')
    plt.plot(moving_average(n2, 2000), label='SARSA')
    plt.plot(moving_average(n3, 2000), label='Q Learning + new state')
    plt.plot(moving_average(n4, 2000), label='SARSA + new state')
    plt.xlabel('Episode Number')
    plt.ylabel('Number of Moves')
    plt.ylim(2,5)
    plt.title('New state representation')
    plt.legend()
    plt.show()

    exit(0)


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


    # Q Learning, Ablation: SGD vs. RMSprop vs. AdamW
    r1 = np.load('results/R_q_AdamW.npy')
    r2 = np.load('results/R_q_default.npy')
    r3 = np.load('results/R_q_RMSprop.npy')

    plt.figure()
    plt.plot(moving_average(r1, 1000), label='AdamW')
    plt.plot(moving_average(r2, 1000), label='SGD')
    plt.plot(moving_average(r3, 1000), label='RMSprop')
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.title('SGD vs. RMSprop vs. AdamW in Q Learning')
    plt.legend()
    plt.show()


    n1 = np.load('results/N_q_AdamW.npy')
    n2 = np.load('results/N_q_default.npy')
    n3 = np.load('results/N_q_RMSprop.npy')

    plt.figure()
    plt.plot(moving_average(n1, 1000), label='AdamW')
    plt.plot(moving_average(n2, 1000), label='SGD')
    plt.plot(moving_average(n3, 1000), label='RMSprop')
    plt.xlabel('Episode Number')
    plt.ylabel('Number of Moves')
    plt.title('SGD vs. RMSprop vs. AdamW in Q Learning')
    plt.legend()
    plt.show()

    # SARSA, Ablation: SGD vs. RMSprop vs. AdamW
    r1 = np.load('results/R_sarsa_AdamW.npy')
    r2 = np.load('results/R_sarsa_default.npy')
    r3 = np.load('results/R_sarsa_RMSprop.npy')

    plt.figure()
    plt.plot(moving_average(r1, 1000), label='AdamW')
    plt.plot(moving_average(r2, 1000), label='SGD')
    plt.plot(moving_average(r3, 1000), label='RMSprop')
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.title('SGD vs. RMSprop vs. AdamW in SARSA')
    plt.legend()
    plt.show()

    n1 = np.load('results/N_sarsa_AdamW.npy')
    n2 = np.load('results/N_sarsa_default.npy')
    n3 = np.load('results/N_sarsa_RMSprop.npy')

    plt.figure()
    plt.plot(moving_average(n1, 1000), label='AdamW')
    plt.plot(moving_average(n2, 1000), label='SGD')
    plt.plot(moving_average(n3, 1000), label='RMSprop')
    plt.xlabel('Episode Number')
    plt.ylabel('Number of Moves')
    plt.title('SGD vs. RMSprop vs. AdamW in SARSA')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    visualize()
