# Setting up a random game
# Assumed that player is always at ball and others random
# Scoring always to left to right

import numpy as np
import matplotlib.pyplot as plt

from highlevel.utility import random_env
from highlevel.plot import plot_game

# Determine the range to sample from
(s_dict, rng) = random_env()

# Plot state of game
plot_game(s_dict)
plt.show(block=False)
plt.pause(0.5)  # hold open for one second
plt.close()

# Plot 
# plt.ion()
rng = np.random.default_rng(seed=1)
for i in range(5):
    (s_dict, _) = random_env(rng=rng)
    plot_game(s_dict)
    # plt.draw()
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()
