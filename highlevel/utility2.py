import numpy as np
import itertools
import random
import time
from numba import njit
import matplotlib.pyplot as plt

import highlevel.constants as K
from highlevel.utility import dist2pts, cart2pol, cart2th, wrap_pn_pi
from highlevel.plot import plot_robot

### Constants
foot_offset = 0.1  # offset from center of the body
MAX_POSES = 200  # Number of solutions to consider

# Field dimensions
field_buffer = 1.0  # Shrinks the dimension of the field to adjust for out of bounds
FL = K.field_length/2
FW = K.field_width/2 - field_buffer

# Goal post dimension
GW = K.goal_width/2
post_rad = 0.3  # Goal post radius to treat like an obstacle
depth_offset = 0.2
goal_depth = K.goal_depth - depth_offset  # depth into goal to avoid going to far back
# TODO: check if depth is 0.6

### Cost function for possible poses
def pose_cost(
        pose_xyw,
        des_pose,
        bxy,
        objxy_list=np.array([[]]),
        c_dist_des=10,  # Cost parameter for the distance to the desired pose
        c_dist_ball=8, # Cost parameter for the distance to the ball
        ):

    # Exclude poses which are behind the ball (don't want to attack own goal)
    # pose_xyw = pose_xyw[:, pose_xyw[0, :] < bxy[0, 0]]

    pose_xywc = np.zeros((pose_xyw.shape[1], 4))
    for i in range(pose_xyw.shape[1]):
        cost_dist_des = - c_dist_des * dist2pts(pose_xyw[:2, i], des_pose)
        cost_dist_ball = - c_dist_ball * dist2pts(pose_xyw[:2, i], bxy)

        cost = cost_dist_des + cost_dist_ball
    
        pose_xywc[i, :3] = pose_xyw[:3, i]
        pose_xywc[i, 3] = cost
    
    # Sort by minimum cost
    idx_area = np.flip(np.argsort(pose_xywc[:, 3]))
    pose_xywc = pose_xywc[idx_area]

    return pose_xywc

### Pose and orientation calculations
def des_poses(
        des_pose,  # x y th position desired pose orientation is into ball
        player_size,  # radius of player
        bxy,  # Ball x y position, shape (2, ),
        ball_size,  # radius of the ball or boundary in which shouldn't be encroached by desired pose
        objxy_list=np.array([[]]),  # Opponents, Objects (x, y, radius)
        d_filter=5,  # Filter obstacles/opponents in solution
        c_pts=100,  # Number of points around circle
        th_range=(-np.pi / 2, np.pi / 2),  # Orientation range to consider
        debug=False,  # Plot figures
        debug_more=False
        ):
    """Returns a list of possible poses if opponent is obstructing"""
    if debug:
        des_xyr = np.append(des_pose, player_size).reshape(1, 3)
        # Desired pose: magenta
        plot_robot(des_xyr, marker='mo', quiver=False, circle=True, circle_pts=200, center=True)
        ball_xyr = np.append(bxy, ball_size).reshape(1, 3)
        # Ball: green
        plot_robot(ball_xyr, marker='go', quiver=False, circle=True, circle_pts=200, center=False)
        
        # Opponents: red
        for obj in objxy_list.T:
            plot_robot(obj.reshape(1, 3), marker='ro', quiver=False, circle=True, circle_pts=200, center=False)
        
        # Goal posts: green
        for gp in goalp_obj.T:
            plot_robot(gp.reshape(1, 3), marker='go', quiver=False, circle=True, circle_pts=200, center=False)
        
        plt.show(block=False)

    # First check if even need to be adjusted
    if objxy_list.size == 0:
            cnstr = np.hstack((np.append(bxy, ball_size).reshape(3, 1), goalp_obj))
    else:
        cnstr = np.hstack((objxy_list, np.append(bxy, ball_size).reshape(3, 1), goalp_obj))
    player = np.append(des_pose, player_size).reshape(3, 1)
    violation = circle_violation(player, cnstr.T)
    if not violation:
        th = cart2th(des_pose - bxy.T[0])
        if th < th_range[0] or th > th_range[1]:
            th += np.pi
        des_xyr = np.append(des_pose, th).reshape(3, 1)
        if debug_more:
            plot_robot(des_xyr.reshape(1, 3), marker='co', quiver=False, circle=True, circle_pts=c_pts, center=True)
            plt.show(block=False)
        return des_xyr

    # Get rid of constraints too far away
    (c_r, _) = cart2pol(cnstr[:2, :] - bxy)

    # Take out objects too far away
    cnstr2 = cnstr[:, c_r < d_filter]

    # Adjust constraints to account for the player size
    cnstr2[2, :] += player_size

    # Get object solutions
    obj_sol = np.zeros((3, c_pts * cnstr2.shape[1]))
    obj_sol[2, :] = player_size
    for i in range(cnstr2.shape[1]):
        obj_sol[:2, i * c_pts: (i + 1) * c_pts] = ring_xy(cnstr2[:2, i], cnstr2[2, i], pts=c_pts)

    if debug_more:
        for i in range(obj_sol.shape[1]):
            plot_robot(obj_sol.T[i].reshape(1, 3), marker='co', quiver=False, circle=True, circle_pts=200, center=True)
        plt.show(block=False)

    # Get rid of solutions in violation of other circles
    count = 0
    sol = np.zeros_like(obj_sol)
    for i in range(obj_sol.shape[1]):
        if not circle_violation(obj_sol[:, i], cnstr.T):
            sol[:, count] = obj_sol[:, i]
            count += 1
    
    if debug_more:
        for i in range(count):
            plot_robot(sol.T[i].reshape(1, 3), marker='ko', quiver=False, circle=True, circle_pts=200, center=True)
        plt.show(block=False)

    # Calculate Orientations
    poses = np.zeros((3, count))
    poses[:2] = sol[:2, :count]
    for i in range(count):

        # TODO: Need to check both solutions
        th = cart2th(poses[:2, i] - bxy.T[0])
        if th < th_range[0] or th > th_range[1]:  # TODO Choose range for how backwards can move with theta range
            th = wrap_pn_pi(th + np.pi)
        poses[2, i] = th

    if debug:
        # Draw the actual arrows
        for i in range(count):
            plot_robot(poses.T[i].reshape(1, 3), marker='ko', quiver=True, circle=False, circle_pts=200, center=True)
        plt.show(block=False)

    # TODO sine similarity filter
        
    return poses

def kick_side(
        pose_xywc,
        bxy,
        th_range=(-np.pi/2, np.pi),
        debug=False
):
    # Calculate solutions with right, left foot and center
    count = 0
    poses2 = np.zeros((3, 3))

    dp = dist2pts(pose_xywc[:2], bxy.T[0])
    thp = np.tan(foot_offset, np.array([dp]))
    poses2[:, count] = pose_xywc[:3]
    count += 1

    thp1 = wrap_pn_pi(pose_xywc[2] + thp)
    if thp1 < th_range[0] or thp1 > th_range[1]:
        # thp1 = wrap_pn_pi(thp1 + np.pi)
        pass
    else:
        poses2[:, count] = np.append(pose_xywc[:2], thp1)
        count += 1

    thp2 = wrap_pn_pi(pose_xywc[2] - thp)
    if thp2 < th_range[0] or thp2 > th_range[1]:
        # thp2 = wrap_pn_pi(thp2 + np.pi)
        pass
    else:
        poses2[:, count] = np.append(pose_xywc[:2], thp2)
        count += 1
    
    poses2 = poses2[:, :count]

    if debug:
        for i in range(count):
            plot_robot(poses2.T[i].reshape(1, 3), marker='co', quiver=True, circle=False, circle_pts=200, center=True)
        plt.show(block=False)
    
    return poses2

### Geometric functions
# @njit
def ring_xy(pos, rad, pts=500):
    th = np.linspace(0, 2 * np.pi, pts + 1)[:-1]
    return np.array([
        rad*np.cos(th) + pos[0], 
        rad*np.sin(th) + pos[1]
        ])

# @njit
def circle_violation(circle, cnstr, eps=0.001):
    """Checks if a circle intersects with any other circles in the constraint list"""
    for c in cnstr:
        d = dist2pts(circle[:2], c[:2])
        if d < (circle[2] + c[2]) - eps:
            return True
    return False

# @njit
def circle_circle(c1, c2, eps=0.0001):
    """Intersections between two circles
    c1 and c2 are x, y, and r
    https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect
    """
    # Distance between radii
    d = dist2pts(c1[:2], c2[:2])

    # Circles are concentric
    if d < eps:
        return (None, False)

    # Distance between c1 and chord formed by intersection points
    l = (c1[2] ** 2 - c2[2] ** 2 + d ** 2) / (2 * d)

    # Circles are not touching and only intersection is in imaginary space
    if l > c1[2]:
        return (None, False)

    # Distance from d line to one of the intersection points
    h = np.sqrt(c1[2] ** 2 - l ** 2)

    # Return if only one solution
    x = l / d * (c2[0] - c1[0]) + c1[0]
    y = l / d * (c2[1] - c1[1]) + c1[1]
    if h < eps:
        return (np.array([[x, y]]), True)
    
    # Else return two solutions
    xp = h / d * (c2[1] - c1[1])
    yp = h / d * (c2[0] - c1[0])
    return (np.array([[x + xp, y - yp], [x - xp, y + yp]]), True)

### Field Constants
# Line segment dictionary
# Note: Kept here so only have to use this utilities 
#       instead of making changes to constants.py
#       constants.py is the same as localization implementation
corners = np.array([
    [-FL, -FW], 
    [ FL, -FW],
    [-FL,  FW],
    [ FL,  FW]
    ]).T

sidelines = np.array([
    [[-FL,-FW], [ FL, -FW]],
    [[ FL, FW], [ FL, -FW]],
    [[-FL, FW], [ FL,  FW]],
    [[-FL, FW], [-FL, -FW]]
    ])

opp_goalp = np.array([
    [FL,  FL], 
    [GW, -GW],
])

goalp_obj = np.array([
    [FL,             FL,      -FL,      -FL], 
    [GW,            -GW,       GW,      -GW],
    [post_rad, post_rad, post_rad, post_rad]
])
