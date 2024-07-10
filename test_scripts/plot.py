# Plotting tools available and how to use them

import numpy as np
import matplotlib.pyplot as plt

from highlevel.plot import plot_field, plot_robot, plot_lines, plot_game
from highlevel.utility import random_env

# Plot field points black circles
plot_field()

# Plot Player (Myself) in Blue
p = np.array([[1, 2, 90*np.pi/180]])  # x, y position and theta orientation
plot_robot(p, marker="bo")

# Plot other Players/Obstacles/Uknowns in Red
opp = np.array([[0, 0, 0.5],
                [4, -1, 2],
                [-3, -1, 0.1],
                [6, 3, 4]])
plot_robot(opp, marker="ro", quiver=False, circle=True)

# Plot Ball
b = np.array([[-3, 4, 0.2]])
plot_robot(b, marker="kx", quiver=False, circle=True)

plot_lines(b[:, :2], opp[:, :2], marker='g-')
plt.show(block=True)


### More Complex Plotting

# Determine the range to sample from
(s_dict, rng) = random_env()

# Plot state of game
plot_game(s_dict)
plt.show(block=True)