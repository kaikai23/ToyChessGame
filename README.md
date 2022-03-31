# Exploiting Q-learning and SARSA to Checkmate a Simplified Chess Game
The assignment of Introduction to Reinforcement Learning. 2022 Spring.

<img width="752" alt="image" src="https://user-images.githubusercontent.com/71677542/161056143-79a2827d-5f82-4f81-9485-990e30c536d4.png">




## How to reproduce the experiments
The rewards and number of moves are stored as `.npy` files in the folder `results/`.

#### To visualize stored experiment results, 

run `Visualize.py`.

#### To rerun all our the experiments, 
run `PlayGame.py`. This will run all the experiments and overwrite files in `results/` folder, and also visualize them in the end. You can also visualize them anytime by running `Visualize.py`. The only thing matters is the files in `results/`.

## Try your own experiment in three lines of code!
In `PlayGame.py`, under `if __name__ == '__main__':`, add your experiment by adding three lines of code.

Example:

<code>agent.reset_all()</code>

<code>agent.set_discount_factor(0.99)</code>

<code>R, N_moves = agent.train_q_learning()</code>

Also don't forget to save them if you want to visualize the results again in the future!
 
## Structure of our code
Our code are in two independent files, `PlayGame.py` and `Visualize.py`. The latter consists of code visualizing our experiments results, while the former consists of code for the main logic.

`PlayGame.py` has 2 classes: `Q_Net`, `Agent` and a function `EpsilonGreedy_Policy`.

The main code is in `Agent`, it has the network and all hyper-parameters as its attributes, and has methods to set different hyperparameters. The most important part of `Agent` is the `train_q_learining` method and `train_sarsa` method, which are the training loops.
## Highlights
1. We construct a clear and neat abstraction of both the problem and our architecture. See **II. METHODS** in our report.
2. We write a highly modularized code, thanks to which we can run all the experiments by excuting `PlayGame.py` only once, and repeatition of code is reduced to minimum. See class **`Agent`** in `PlayGame.py`.
3. We did ablation study for each hyper-parameter that is in concern for better analyzing the results. See **III. EXPERIMENTS** in our report.

## Report
see `report.pdf`.
