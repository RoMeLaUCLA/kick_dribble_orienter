import numpy as np
import matplotlib.pyplot as plt

import highlevel.constants as K

def plot_field():
    """Adds the field features to a pyplot"""
    plt.plot(K.field_pts[:, 1], K.field_pts[:, 2], 'ko' ,markerfacecolor="none")
    plt.axis('equal')
    return None

def plot_robot(robot, marker='bo', quiver=True, circle=False, circle_pts=200, center=True):
    """Plots robot in pyplot"""
    x = robot[:, 0]
    y = robot[:, 1]
    if center:
        plt.plot(x, y, marker)
    if quiver:
        theta = robot[:, 2]
        plt.quiver(x, y, np.cos(theta), np.sin(theta), color=marker[0])
    if circle:
        for i in range(robot.shape[0]):
            x = robot[i, 0]
            y = robot[i, 1]
            rad = robot[i, -1]
            (x, y) = circle_xy((x, y), rad, circle_pts)
            plt.plot(x, y, marker, markersize=0.2)
    return None

def circle_xy(pos, rad, pts=500):
    th = np.linspace(0, 2 * np.pi, pts)
    x  = rad*np.cos(th) + pos[0]
    y  = rad*np.sin(th) + pos[1]
    return (x, y)

def plot_lines(xy_root, xy_branches, marker='g-'):
    assert xy_branches.shape[0] >= xy_branches.shape[1]

    for i in range(xy_branches.shape[0]):
        plt.plot([xy_root[0, 0], xy_branches[i, 0]], 
                 [xy_root[0, 1], xy_branches[i, 1]], marker)
    return None
        
def plot_game(state_dict):
    plot_field()

    # Player (self)
    # plot_robot(state_dict["t1"], marker="bo")
    plot_robot(state_dict["t1"], marker="bo", quiver=False, circle=True)
    t1_q = state_dict["t1"].copy()
    t1_q[0, 2] = 0.0
    plot_robot(t1_q, marker="bo")

    # Teammate
    plot_robot(state_dict["t2"], marker="bo", quiver=False, circle=True)

    # Opposing team
    plot_robot(state_dict["op1"], marker="ro", quiver=False, circle=True)
    plot_robot(state_dict["op2"], marker="ro", quiver=False, circle=True)

    # Plot Ball
    plot_robot(state_dict['b'], marker="kx", quiver=False, circle=True)
    return None

### Plot tools from localization

def plot_obs(obs, robot):
    plt.plot(obs[:, 0], obs[:, 1], 'g.')
    plt.plot([robot[0], -np.sin(robot[2])*10 + robot[0]], [robot[1],  np.cos(robot[2])*10 + robot[1]], 'g--')
    plt.plot([robot[0],  np.sin(robot[2])*10 + robot[0]], [robot[1], -np.cos(robot[2])*10 + robot[1]], 'g--')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')