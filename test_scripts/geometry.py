# Building up geometric functions needed to determine areas to kick

import numpy as np
import time
import matplotlib.pyplot as plt

import highlevel.constants as K
from highlevel.utility import random_env, ball_rays, circle_cost, plot_robot
from highlevel.plot import plot_game

N = 100
seed = 0

t = 0
rng = np.random.default_rng(seed=seed)
for i in range(N):

    # Get state of  game
    (s_dict, _) = random_env(rng=rng)

    # Debug
    # s_dict["b"][:2] = np.array([[6.744, -1.544]])
    # nu = rng.uniform(low=-2.0, high=2.0, size=(2, ))
    # s_dict["op1"][0, :2] = s_dict["b"][0, :2] + nu
    # nu1 = rng.uniform(low=-2.0, high=2.0, size=(2, ))
    # s_dict["op2"][0, :2] = s_dict["b"][0, :2] + nu1

    # Plot the game
    plot_game(s_dict)

    # Range of angle of kicks to consider assuming straight ahead is zero angle
    th_range = np.array([-np.pi/2 - 0.1, np.pi/2 + 0.1])

    # This planner is relative to t1
    # Convert to x, y relative to players and create a sorted list to iterate through
    t1  = s_dict["t1"][0, :2]  # t1 is yourself
    t2  = s_dict["t2"][0, :2]
    op1 = s_dict["op1"][0, :2]
    op2 = s_dict["op2"][0, :2]

    # TODO: Implement a rotation matrix to flip sides of field

    # Formatting has to be in 2 x n axis
    b = np.array([s_dict['b'][0, :2]]).T
    t_list = np.stack((t1, t2), axis=1)
    opp_list = np.stack((op1, op2), axis=1)

    # First iteration of paths to kick
    # Note: last element is radius squared so area component
    tic = time.time()
    circles_xysqr = ball_rays(
        b, 
        objxy_list=opp_list, 
        th_range=th_range, 
        timeout=None,
        beta=0.1,
        alpha=0.3,
        obj_r=0.4,
        closer_off=0.2,
        add_closer=True,
        debug=True,
        debug_more=False,
        debug_closer=False
        )
    if i != 0:
        tictoc = time.time() - tic
        t += tictoc
        # print(f"Time: {tictoc:.3f}s")

    # TODO: Add center goal circles later maybe?

    # # Cost function for largest area circle
    # circlesxysqr_cost = circle_cost(
    #     b, 
    #     circles_xysqr,
    #     txy_list=t_list, 
    #     oppxy_list=opp_list,
    #     th_goal=0.,  # cost parameter associated with how large the angle to goal is
    #     area=10.,  # cost parameter associated with r squared of circle so area
    #     b_goal=0,  # cost parameter associated with ball distance to goal
    #     opp_lam=0.,  # cost parameter associated with opponents blocking kicking path
    #     lam1=0.,  # exponential decay parameter for opp_lam
    #     t_self=0.,  # cost parameter associated with self orienation
    #     t_lam=0.,  # cost parameter associated with orientation of teammates and self to desired
    #     lam2=0,  # exponential decay parameter for t_lam
    #     opp_beta=0.,  # cost parameter associated with opponents distance to ball
    #     lam3=0,  # exponential decay parameter for opp_beta
    #     tb_self=0., # cost parameter associated with self distance to circle center
    #     t_beta=0.,  # cost parameter associated with teammates and self distance to circle center
    #     lam4=0,  # exponential decay parameter for t_beta
    #     timeout=None,
    #     debug=False
    #     )


    # Cost function for good angle and distance to goal
    circlesxysqr_cost = circle_cost(
        b, 
        circles_xysqr,
        txy_list=t_list, 
        oppxy_list=opp_list,
        th_goal=5.0,  # cost parameter associated with how large the angle to goal is
        area=0.1,  # cost parameter associated with r squared of circle so area
        b_goal=0.2,  # cost parameter associated with ball distance to goal
        opp_lam=0.0,  # cost parameter associated with opponents blocking kicking path
        lam1=8,  # exponential decay parameter for opp_lam
        t_self=0.,  # cost parameter associated with self orienation
        t_lam=0.,  # cost parameter associated with orientation of teammates and self to desired
        lam2=8,  # exponential decay parameter for t_lam
        opp_beta=0.,  # cost parameter associated with opponents distance to ball
        lam3=8,  # exponential decay parameter for opp_beta
        tb_self=0., # cost parameter associated with self distance to circle center
        t_beta=0.,  # cost parameter associated with teammates and self distance to circle center
        lam4=8,  # exponential decay parameter for t_beta
        timeout=None,  # break loop if it takes too long but atleast find one solution
        debug=False  # plots best circle
        )
    
    # # Cost function for gaining distance and angle towards goal but kickable
    # circlesxysqr_cost = circle_cost(
    #     b, 
    #     circles_xysqr,
    #     txy_list=t_list, 
    #     oppxy_list=opp_list,
    #     th_goal=0.8,  # cost parameter associated with how large the angle to goal is
    #     area=2.0,  # cost parameter associated with r squared of circle so area
    #     b_goal=1.5,  # cost parameter associated with ball distance to goal
    #     opp_lam=0.0,  # cost parameter associated with opponents blocking kicking path
    #     lam1=8,  # exponential decay parameter for opp_lam
    #     t_self=0.,  # cost parameter associated with self orienation
    #     t_lam=0.,  # cost parameter associated with orientation of teammates and self to desired
    #     lam2=8,  # exponential decay parameter for t_lam
    #     opp_beta=0.,  # cost parameter associated with opponents distance to ball
    #     lam3=8,  # exponential decay parameter for opp_beta
    #     tb_self=0., # cost parameter associated with self distance to circle center
    #     t_beta=0.,  # cost parameter associated with teammates and self distance to circle center
    #     lam4=8,  # exponential decay parameter for t_beta
    #     timeout=None,  # break loop if it takes too long but atleast find one solution
    #     debug=False  # plots best circle
    #     )
       
    # # Cost function for kick with least correction needed in orientation to ball ie draw line straight through ball
    # circlesxysqr_cost = circle_cost(
    #     b, 
    #     circles_xysqr,
    #     txy_list=t_list, 
    #     oppxy_list=opp_list,
    #     th_goal=0.,  # cost parameter associated with how large the angle to goal is
    #     area=0.0,  # cost parameter associated with r squared of circle so area
    #     b_goal=0.,  # cost parameter associated with ball distance to goal
    #     opp_lam=0.0,  # cost parameter associated with opponents blocking kicking path
    #     lam1=8,  # exponential decay parameter for opp_lam
    #     t_self=5.,  # cost parameter associated with self orienation
    #     t_lam=0.,  # cost parameter associated with orientation of teammates and self to desired
    #     lam2=8,  # exponential decay parameter for t_lam
    #     opp_beta=0.,  # cost parameter associated with opponents distance to ball
    #     lam3=8,  # exponential decay parameter for opp_beta
    #     tb_self=0., # cost parameter associated with self distance to circle center
    #     t_beta=0.,  # cost parameter associated with teammates and self distance to circle center
    #     lam4=8,  # exponential decay parameter for t_beta
    #     timeout=None,  # break loop if it takes too long but atleast find one solution
    #     debug=False  # plots best circle
    #     )
    
    # # Avoid circles where defenders can block kicking path
    # circlesxysqr_cost = circle_cost(
    #     b, 
    #     circles_xysqr,
    #     txy_list=t_list, 
    #     oppxy_list=opp_list,
    #     th_goal=0.,  # cost parameter associated with how large the angle to goal is
    #     area=0.0,  # cost parameter associated with r squared of circle so area
    #     b_goal=0.,  # cost parameter associated with ball distance to goal
    #     opp_lam=10.0,  # cost parameter associated with opponents blocking kicking path
    #     lam1=8,  # exponential decay parameter for opp_lam
    #     t_self=0.0,  # cost parameter associated with self orienation
    #     t_lam=0.,  # cost parameter associated with orientation of teammates and self to desired
    #     lam2=8,  # exponential decay parameter for t_lam
    #     opp_beta=0.,  # cost parameter associated with opponents distance to ball
    #     lam3=8,  # exponential decay parameter for opp_beta
    #     tb_self=0., # cost parameter associated with self distance to circle center
    #     t_beta=0.,  # cost parameter associated with teammates and self distance to circle center
    #     lam4=8,  # exponential decay parameter for t_beta
    #     timeout=None,  # break loop if it takes too long but atleast find one solution
    #     debug=False  # plots best circle
    #     )

    circle_xyr = circlesxysqr_cost[0, :3].copy()
    circle_xyr[2] = np.sqrt(circle_xyr[2])
    plot_robot(circle_xyr.reshape((1, 3)), marker="bo", quiver=False, circle=True)
    
    # # Avoid distance to defenders
    # circlesxysqr_cost = circle_cost(
    #     b, 
    #     circles_xysqr,
    #     txy_list=t_list, 
    #     oppxy_list=opp_list,
    #     th_goal=0.,  # cost parameter associated with how large the angle to goal is
    #     area=0.0,  # cost parameter associated with r squared of circle so area
    #     b_goal=0.,  # cost parameter associated with ball distance to goal
    #     opp_lam=0.,  # cost parameter associated with opponents blocking kicking path
    #     lam1=8,  # exponential decay parameter for opp_lam
    #     t_self=0.0,  # cost parameter associated with self orienation
    #     t_lam=0.,  # cost parameter associated with orientation of teammates and self to desired
    #     lam2=8,  # exponential decay parameter for t_lam
    #     opp_beta=10.,  # cost parameter associated with opponents distance to ball
    #     lam3=8,  # exponential decay parameter for opp_beta
    #     tb_self=0., # cost parameter associated with self distance to circle center
    #     t_beta=0.,  # cost parameter associated with teammates and self distance to circle center
    #     lam4=8,  # exponential decay parameter for t_beta
    #     timeout=None,  # break loop if it takes too long but atleast find one solution
    #     debug=False  # plots best circle
    #     )
    
    # # Pass to teammate
    # circlesxysqr_cost = circle_cost(
    #     b, 
    #     circles_xysqr,
    #     txy_list=t_list, 
    #     oppxy_list=opp_list,
    #     th_goal=0.,  # cost parameter associated with how large the angle to goal is
    #     area=0.0,  # cost parameter associated with r squared of circle so area
    #     b_goal=0.,  # cost parameter associated with ball distance to goal
    #     opp_lam=0.,  # cost parameter associated with opponents blocking kicking path
    #     lam1=8,  # exponential decay parameter for opp_lam
    #     t_self=0.0,  # cost parameter associated with self orienation
    #     t_lam=0.,  # cost parameter associated with orientation of teammates and self to desired
    #     lam2=8,  # exponential decay parameter for t_lam
    #     opp_beta=0.,  # cost parameter associated with opponents distance to ball
    #     lam3=8,  # exponential decay parameter for opp_beta
    #     tb_self=0., # cost parameter associated with self distance to circle center
    #     t_beta=10.,  # cost parameter associated with teammates and self distance to circle center
    #     lam4=8,  # exponential decay parameter for t_beta
    #     timeout=None,  # break loop if it takes too long but atleast find one solution
    #     debug=False  # plots best circle
    #     )
    
    # # Dribble or go through ball
    # circlesxysqr_cost = circle_cost(
    #     b, 
    #     circles_xysqr,
    #     txy_list=t_list, 
    #     oppxy_list=opp_list,
    #     th_goal=0.,  # cost parameter associated with how large the angle to goal is
    #     area=0.0,  # cost parameter associated with r squared of circle so area
    #     b_goal=0.04,  # cost parameter associated with ball distance to goal
    #     opp_lam=1.,  # cost parameter associated with opponents blocking kicking path
    #     lam1=8,  # exponential decay parameter for opp_lam
    #     t_self=2.5,  # cost parameter associated with self orienation
    #     t_lam=0.,  # cost parameter associated with orientation of teammates and self to desired
    #     lam2=8,  # exponential decay parameter for t_lam
    #     opp_beta=3.,  # cost parameter associated with opponents distance to circle center
    #     lam3=8,  # exponential decay parameter for opp_beta
    #     tb_self=0.1, # cost parameter associated with self distance to circle center
    #     t_beta=0.,  # cost parameter associated with teammates and self distance to circle center
    #     lam4=8,  # exponential decay parameter for t_beta
    #     timeout=None,  # break loop if it takes too long but atleast find one solution
    #     debug=False  # plots best circle
    #     )

    circle_xyr = circlesxysqr_cost[0, :3].copy()
    circle_xyr[2] = np.sqrt(circle_xyr[2])
    plot_robot(circle_xyr.reshape((1, 3)), marker="bo", quiver=False, circle=True)

    plt.show(block=True)
    plt.xlim(-10, 10) 
    plt.ylim(-5, 5)
    plt.close()

# Note Numba on first call takes up to 4 sec
print(f"Avg. time: {t/(N - 1):.3f} s")
print(f"Avg. Hz: {(N - 1)/t:.0f} hz")

# TODO: Apply this function recursively to get a solution probably only two steps but hardware has to work first

