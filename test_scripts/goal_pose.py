# Functionality for adjusting based on opponents

import numpy as np
import time
import matplotlib.pyplot as plt

import highlevel.constants as K
from highlevel.utility import random_env, plot_robot
from highlevel.utility2 import des_poses, pose_cost, kick_side
from highlevel.plot import plot_game

N = 100
seed = 0
nu = 1.0

t = 0
rng = np.random.default_rng(seed=seed)
for i in range(N):

    # Get state of  game
    (s_dict, _) = random_env(rng=rng)

    # Place players in close proximity of ball
    nu0 = rng.uniform(low=-nu, high=nu, size=(2, ))
    s_dict["op1"][0, :2] = s_dict["b"][0, :2] + nu0
    nu1 = rng.uniform(low=-nu, high=nu, size=(2, ))
    s_dict["op2"][0, :2] = s_dict["b"][0, :2] + nu1
    nu2 = rng.uniform(low=-nu, high=nu, size=(2, ))
    s_dict["t1"][0, :2] = s_dict["b"][0, :2] + nu2
    nu3 = rng.uniform(low=[-nu, -nu], high=[0, nu], size=(2, ))
    des_pose = s_dict["b"][0, :2] + nu3

    # Plot the game
    plot_game(s_dict)
    plt.show(block=False)

    # This planner is relative to t1
    # Convert to x, y relative to players and create a sorted list to iterate through
    t1  = s_dict["t1"][0, :2]  # t1 is yourself
    t2  = s_dict["t2"][0, :2]
    op1 = s_dict["op1"][0, :3]
    op2 = s_dict["op2"][0, :3]

    # TODO: Implement a rotation matrix to flip sides of field

    # Formatting has to be in 2 x n axis
    b = np.array([s_dict['b'][0, :2]]).T
    t_list = np.stack((t1, t2), axis=1)
    opp_list = np.stack((op1, op2), axis=1)

    # Sizes or safety ring
    player_size = 0.35  # radius of ring around t1 or self
    ball_size = 0.3  # radius of ball or ring around it

    # Generate all desired poses
    tic = time.time()
    poses = des_poses(
        des_pose,  # x y th position desired pose orientation is into ball
        player_size,  # radius of player
        b,  # Ball x y position, shape (2, ),
        ball_size,  # radius of the ball or boundary in which shouldn't be encroached by desired pose
        objxy_list=opp_list,  # Opponents, Objects (x, y, radius)
        d_filter=2,  # Filter obstacles/opponents in solution
        c_pts=50,  # Number of points around circle
        th_range=(-np.pi / 2, np.pi / 2),  # Orientation range to consider
        # WARNING: abs th_range shouldn't go below pi/2 or be unsymmetric otherwise issues
        debug=True,  # Plot figures
        debug_more=False
        )
    if i != 0:
        tictoc = time.time() - tic
        t += tictoc
        print(f"Hz: {1/tictoc:.0f}")

    # Cost function
    pose_xywc = pose_cost(
        poses,
        des_pose,
        b,
        objxy_list=opp_list,
        filter_dribble=True
    )

    # Desired pose: magenta
    plot_robot(np.append(des_pose, player_size).reshape(1, 3), marker='mo', quiver=False, circle=True, circle_pts=200, center=True)
    # Ball: green
    plot_robot(np.append(b, ball_size).reshape(1, 3), marker='go', quiver=False, circle=True, circle_pts=200, center=False)
    # Selected pose arrow: cyan quiver
    plot_robot(pose_xywc[0, :3].reshape(1, 3), marker='co', quiver=True, circle=False, circle_pts=200, center=True)
    # Selected pose: cyan circle
    plot_robot(np.append(pose_xywc[0, :2], player_size).reshape(1, 3), marker='co', quiver=False, circle=True, circle_pts=200, center=False)

    sol = pose_xywc[0]
    kicks = kick_side(
        sol,
        b,
        th_range=(-np.pi/2, np.pi),
        debug=True
        )

    plt.show(block=True)
    plt.xlim(-10, 10) 
    plt.ylim(-5, 5)
    plt.close()
