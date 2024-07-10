import numpy as np
import itertools
import random
import time
import matplotlib.pyplot as plt
from numba import njit

import highlevel.constants as K
from highlevel.plot import plot_robot


### Constants
# Do NOT change minimum rays without consulting
MIN_RAYS = 2  # Minimum number of boundaries considered without obstacles/opponents
MAX_RAYS = 12  # Maximum number including boundaries and obstacles/opponents

# Field dimensions
field_buffer = 1.0  # Shrinks the dimension of the field to adjust for out of bounce
FL = K.field_length/2
FW = K.field_width/2 - field_buffer

# Goal post dimension
GW = K.goal_width/2

# Random Env
field_extender = field_buffer + 1.0 # To simulate throw in options

### Cost Function
def circle_cost(
        bxy,  # x, y position of ball
        circlesxysqr,  # circles x, y, and sqaured of r
        txy_list=np.array([[]]), # x, y position of self and teammate in that order
        oppxy_list=np.array([[]]),  # x, y opponent list
        th_goal=0.3,  # cost parameter associated with how large the angle to goal is
        area=0.1,  # cost parameter associated with r squared of circle so area
        b_goal=0.8,  # cost parameter associated with ball distance to goal
        opp_lam=0.4,  # cost parameter associated with opponents blocking kicking path
        lam1=8,  # exponential decay parameter for opp_lam
        t_self=0.2,  # cost parameter associated with self orienation
        t_lam=0.2,  # cost parameter associated with orientation of teammates and self to desired
        lam2=8,  # exponential decay parameter for t_lam
        opp_beta=0.4,  # cost parameter associated with opponents distance to circle center
        lam3=8,  # exponential decay parameter for opp_beta
        tb_self=0.2, # cost parameter associated with self distance to circle center
        t_beta=0.2,  # cost parameter associated with teammates and self distance to circle center
        lam4=8,  # exponential decay parameter for t_beta
        timeout=None,  # break loop if it takes too long but atleast find one solution
        debug=False  # plots best circle
        ):
    """Cost function for circles for kicking
    Note: Lower cost is better
    Warning: All parameters given related to cost parameters are positive!
    """
    # Breakout early if needed
    tic = time.time()

    count = 0
    shp = circlesxysqr.shape 
    circlesxysqr_cost = np.zeros((shp[0], shp[1] + 1))
    for circle in circlesxysqr:
        cost = 0

        # Area of circle radius squared (Positive Cost)
        csqr = circle[2]
        cost_area = area * csqr
        
        # Angle to goal posts (Positive Cost)
        cxy = circle[:2]
        th1 = cart2th(opp_goalp[:, 0] - cxy)
        th2 = cart2th(opp_goalp[:, 1] - cxy)
        th = np.abs(th1 - th2)
        cost_goalth = th_goal * th

        # Distance to center of goal (Negative Cost)
        bg_dist = dist2pts(cxy, np.array([FL, 0.0]))
        cost_goald = - b_goal * bg_dist

        # Angle needed to block the kick trajectory needed by opponents (Negative Cost)
        # TODO: Add arc length so need radius
        (opp_r, opp_th) = cart2pol(oppxy_list - bxy)
        b_th = cart2th(cxy[:2] - bxy.T[0])
        cost_oppth = 0
        for i in range(opp_th.shape[0]):
            cost_oppth -= opp_lam / (np.abs(wrap_pn_pi(b_th - opp_th[i])) * opp_r[i] + 1) ** lam1

        # Distance to goal pose closest teammate including self (Inverse Positive)
        (t_r, t_th) = cart2pol(txy_list - bxy)

        # For self
        cost_thself = t_self / (np.abs(wrap_pn_pi(b_th + np.pi - t_th[0])) * t_r[0] + 1) ** lam2

        # For others
        cost_tth = 0
        for i in range(1, t_th.shape[0]):
            cost_tth += t_lam / (np.abs(wrap_pn_pi(b_th - t_th[i])) * t_r[i] + 1) ** lam2

        # Circle center distance to opponent (Inverse Negative)
        cost_oppb = 0
        for xy in oppxy_list.T:
            cost_oppb -= opp_beta / (dist2pts(cxy, xy) + 1) ** lam3
        
        # Circle center distance to teammate including self (Inverse Positive)
        # For self
        cost_tbself = tb_self / (dist2pts(cxy, txy_list.T[0]) + 1) ** lam4

        # For others      
        cost_tb = 0
        for i in range(1, txy_list.T.shape[0]):
            cost_tb += t_beta / (dist2pts(cxy, txy_list.T[i]) + 1) ** lam4

        # Add to return container
        circlesxysqr_cost[count, :3] = circle
        cost = cost_area + cost_goald + cost_goalth + cost_oppth + cost_tth + cost_oppb + cost_tb + cost_thself + cost_tbself
        circlesxysqr_cost[count, 3] = cost
        count += 1

        # Breakout of loop if taking too much time
        if timeout is not None and time.time() - tic > timeout:
            circlesxysqr_cost = circlesxysqr_cost[:count]
            break

        if debug:
            circle_xyr = circlesxysqr_cost[count - 1, :3].copy()
            circle_xyr[2] = np.sqrt(circle_xyr[2])
            plot_robot(circle_xyr.reshape((1, 3)), marker="bo", quiver=False, circle=True)
            plt.show(block=False)

    # Sort by minimum cost
    idx_area = np.flip(np.argsort(circlesxysqr_cost[:, 3]))
    circlesxysqr_cost = circlesxysqr_cost[idx_area]
    
    # TODO: Add something that has shortest orientation
    # Add something if something is immediately in path

    return circlesxysqr_cost


### Path Finder
def ball_rays(
        bxy,  # Ball x y position, shape (2, ),
        objxy_list=np.array([]),  # Opponents, Objects (Teammate is always first), shape (2, N)
        th_range=np.array([-np.pi/2, np.pi/2]),  # Range to prevent going backwards depending on where on field, between (-pi, pi)
        rho=5,  # Mulplicative scaling factor to search for line intersections usually at least want 1 but at most probably 5
        alpha=0.8, # Alpha norm difference to filter out similar solutions (be careful not too large because very naive clustering)
        beta=0.5, # Minimum sqaured radius of circle assuming at least 2 circles
        obj_r=0.75,  # Radius of objects in meters 
        timeout=None,  # Break loop if it takes too long but atleast find one solution
        warnings=True,  # Print warnings
        debug=False,  # Plot figures
        debug_more=False,  # Plots more circles
        add_closer=True, # Add Closer Circles To Ensure You Beat Opponents to Ball
        closer_off=0.2,  # Meter offset given to player when calculating closer circles
        debug_closer=False,  # Add circle plots for closer circles
        ):
    """Calculates line segment pairs for making circles to kick to"""
    # Breakout early if needed
    tic = time.time()

    # Ensure objects aren't greater than max rays
    if objxy_list.size > 1 and objxy_list.shape[1] + 4 > MAX_RAYS:
        if warnings:
            print(f"Warning: Maximum number of obstacles exceeds allowable size of {MAX_RAYS} in planner!")
            print("Removing further obstacles! Watch out!")

        idx_dist = np.argsort(dist2pts(objxy_list, bxy))
        objxy_list = objxy_list[idx_dist[:MAX_RAYS]]

    # Field corners to avoid no solution edge case
    # th_range is perpendicular and no triangles are formed with no enemies
    obj_opp_goalp = opp_goalp - bxy
    # obj_opp_goalp = corners - bxy

    # If nothing is passed into obj_ballframe
    if objxy_list.size == 0:
        # Ball reference frame
        obj_ballframe = None

        # Polar theta
        obj_th = cart2th(obj_opp_goalp)
    else:
        # Ball reference frame
        obj_ballframe = (objxy_list - bxy).reshape((2, -1))

        # Change to polar (only need theta for now)
        obj_th = cart2th(np.hstack((obj_ballframe, obj_opp_goalp)))        

    # Ensure angles are in
    th_range = wrap_pn_pi(th_range)

    # TODO: Add vertical line at each player to get solutions under players but maybe not good idea if cannot control power
    #       maybe this is not a good idea
    # Filter any rays outside of theta range
    obj_th = obj_th[obj_th > th_range[0]]
    obj_th = obj_th[obj_th < th_range[1]]

    # Append limits and choose arbitrary rho
    obj_th = np.hstack((np.array([th_range[0], th_range[1]]), obj_th))

    # Convert to points on field
    bxy = bxy.T[0]
    rays = np.zeros((obj_th.size, 2, 2))
    for i in range(obj_th.size):
        rays[i, :, :] = np.array([
            bxy, 
            bxy + 10 * np.array([np.cos(obj_th[i]), np.sin(obj_th[i])])
            ])
    
    # Iterate slice points
    count = 0
    circles = np.zeros((len(combinations_dict[obj_th.size]), 3))
    for comb in combinations_dict[obj_th.size]:

        # Get intersections between different lines
        pt1 = rays[comb[0]] if comb[0] < obj_th.size else sidelines[comb[0] - obj_th.size]
        pt2 = rays[comb[1]] if comb[1] < obj_th.size else sidelines[comb[1] - obj_th.size]
        (pt_a, success) = inter_rayline(
            pt1[0], pt1[1], pt2[0], pt2[1], 
            tmin=-rho, tmax=rho, smin=-rho, smax=rho
            )
        if not success:
            continue

        pt3 = rays[comb[2]] if comb[2] < obj_th.size else sidelines[comb[2] - obj_th.size]
        (pt_b, success) = inter_rayline(
            pt2[0], pt2[1], pt3[0], pt3[1], 
            tmin=-rho, tmax=rho, smin=-rho, smax=rho
            )
        if not success:
            continue

        (pt_c, success) = inter_rayline(
            pt3[0], pt3[1], pt1[0], pt1[1], 
            tmin=-rho, tmax=rho, smin=-rho, smax=rho
            )
        if not success:
            continue

        # Get circle coordinates and radius
        (circle_xysqr, success) = tri_incenter(pt_a, pt_b, pt_c)
        if not success:
            continue

        # Filter out if outside of field
        if abs(circle_xysqr[0]) > FL or abs(circle_xysqr[1]) > FW:
            continue

        # Filter out circles outside of th_range
        circle_th = cart2th(circle_xysqr[:2] - bxy)        
        if circle_th < th_range[0] or circle_th > th_range[1]:
            continue

        if debug:
            plt.plot(pt1[:, 0], pt1[:, 1])
            plt.plot(pt2[:, 0], pt2[:, 1])
            plt.plot(pt3[:, 0], pt3[:, 1])
            plt.plot(pt_a[0], pt_a[1], 'go')
            plt.plot(pt_b[0], pt_b[1], 'go')
            plt.plot(pt_c[0], pt_c[1], 'go')
            if debug_more:
                circle_xyr = circle_xysqr.copy()
                circle_xyr[2] = np.sqrt(circle_xyr[2])
                plot_robot(circle_xyr.reshape((1, 3)), marker="co", quiver=False, circle=True)
            plt.show(block=False)

        
        # Adjust squared radius for field boundaries
        circle_xysqr[2] = min_sqr(sidelines, circle_xysqr)

        if debug_more:
            circle_xyr = circle_xysqr.copy()
            circle_xyr[2] = np.sqrt(circle_xyr[2])
            plot_robot(circle_xyr.reshape((1, 3)), marker="mo", quiver=False, circle=True)
            plt.show(block=False)

        # Adjust squared radius to closest boundary from objects/opponents and th_range
        circle_xysqr[2] = min_sqr(rays[:-2], circle_xysqr)  # Don't include corner ones for edge cases

        if debug_more:
            circle_xyr = circle_xysqr.copy()
            circle_xyr[2] = np.sqrt(circle_xyr[2])
            plot_robot(circle_xyr.reshape((1, 3)), marker="ko", quiver=False, circle=True)
            plt.show(block=False)
        
        # Save circle
        circles[count, :] = circle_xysqr
        count += 1

        # Breakout of loop if taking too much time
        if timeout is not None and time.time() - tic > timeout:
            break
    
    # Add additional circles that are closer to the ball that says we are closer
    if add_closer:
        
        count_a = 0
        circles_a = np.zeros_like(circles)
        for c in circles[:count]:

            # Find closest player to circle
            closest = None
            d_min = np.inf
            for i in range(objxy_list.T.shape[0]):
                d = dist2pts(objxy_list.T[i], c[:2])
                if d < d_min:
                    d_min = d
                    closest = objxy_list.T[i]
            
            # Go onto next circle
            if closest is None:
                continue

            (r, th) = cart2pol(closest[:2] - bxy)
            (r1, th1) = cart2pol(c[:2] - bxy)

            # 0.5 half of isosceles triangles then abs cos subtract a slight offset
            h = np.maximum(0.5 * r / np.abs(np.cos(th1 - th)) - closer_off, 0.0)

            # Solution is beyond maximum solution
            if h > r1:
                continue

            xy = pol2cart(th1, h) + bxy

            circle_xysqr = c.copy()
            circle_xysqr[:2] = xy

            # Filter out if outside of field
            if abs(circle_xysqr[0]) > FL or abs(circle_xysqr[1]) > FW:
                continue

            # Adjust squared radius for field boundaries
            circle_xysqr[2] = min_sqr(sidelines, circle_xysqr)

            # Adjust squared radius to closest boundary from objects/opponents and th_range
            circle_xysqr[2] = min_sqr(rays[:-2], circle_xysqr)  # Don't include corner ones for edge cases

            # Add circles to successful ones
            circles_a[count_a] = circle_xysqr
            count_a += 1

            if debug_closer:
                circle_xyr = circle_xysqr.copy()
                circle_xyr[2] = np.sqrt(circle_xyr[2])
                plot_robot(circle_xyr.reshape((1, 3)), marker="mo", quiver=False, circle=True)
                plt.show(block=False)

        # Stack Solutions
        circles = np.vstack((circles[:count], circles_a[:count_a]))
        count = circles.shape[0]

        # TODO: Add in goal line intersection
        
    # Adjust size for the robot
    circles = circle_opp(circles, objxy_list.T, opp_r=obj_r)

    if debug_more:
        for i in range(circles.shape[0]):
            circle_xyr = circles[i].copy()
            circle_xyr[2] = np.sqrt(circle_xyr[2])
            plot_robot(circle_xyr.reshape((1, 3)), marker="co", quiver=False, circle=True)
            plt.show(block=False)

    # Sort list based on radius squared
    idx_area = np.flip(np.argsort(circles[:, 2]))
    circles = circles[idx_area]

    # Eliminate similar solutions
    count2 = 0
    circles2 = np.zeros((count, 3))
    for i in range(count):
        # Skip seen elements
        if circles[i, 2] == 0.0:
            continue
        
        # Find similar ones
        diff = circles[i, :2] - circles[i:count, :2]
        idx_sim = np.sum(diff * diff, axis=1) < alpha

        # Get max area of similar ones
        circles2[count2] = circles[i]
        count2 += 1

        # Erase ones checked
        circles[i:count][idx_sim] = np.zeros((1, 3))
            
        if debug_more:
            circle_xyr = circles2[count2 - 1].copy()
            circle_xyr[2] = np.sqrt(circle_xyr[2])
            plot_robot(circle_xyr.reshape((1, 3)), marker="ko", quiver=False, circle=True)
            plt.show(block=False)

    # Eliminate too small squared radius
    if count2 >= 1:
        circles3 = np.vstack((circles2[0], circles2[1:count2][circles2[1:count2, 2] > beta]))
    else:
        if warnings:
            print("No valid solutions found! Returning all zeros! By the way this should theoretically never happen so there's a bug or bad inputs like th_range too small!")
        circles3 = np.zeros((1, 3))

    if debug:
        for i in range(circles3.shape[0]):
            circle_xyr = circles3[i].copy()
            circle_xyr[2] = np.sqrt(circle_xyr[2])
            plot_robot(circle_xyr.reshape((1, 3)), marker="mo", quiver=False, circle=True)
            plt.show(block=False)

    return circles3

# # @njit
# def split_circle(lines, circle_xysqr):
#     """Get resized circle with center"""
#     circle = circle_xysqr.copy()
#     for line in lines:
        
#         # Center circle on origin then get intersections
#         lxy = line - circle[:2]
#         (inters, success) = circle_line(lxy[0], lxy[1], circle[2])

#         if not success:
#             continue

#         # Only interested in cases with 2 intersections
#         if inters.shape[0] == 1:
#             continue

#         # Transform back into original frame
#         intersects = inters + circle[:2]
        
#         mid_pt = (intersects[0] + intersects[1]) / 2
#         r = np.sqrt(circle[2]) + dist2pts(mid_pt, circle[:2])

#         th = np.arctan2(circle[1] - mid_pt[1], circle[0] - mid_pt[0])
#         v = rot2d(th) @ np.array([r, 0.])

#         circle[:2] = mid_pt + v
#         circle[2] = r ** 2
    
#     return circle    

# @njit
def circle_opp(circles, opps, opp_r=0.4):
    """Adjust circle size based on present opponents"""
    for i in range(circles.shape[0]):
        for opp in opps:
            d = dist2pts(circles[i, :2], opp[:2])
            r = np.sqrt(circles[i, 2])
            r2 = d - opp_r

            # In the case that it overlaps with point
            if r2 < 0:
                circles[i, 2] = 0.0
                continue
            
            # Adjust radius squared
            if r2 < r:
                circles[i, 2] = r2 ** 2
    
    return circles

# @njit
def min_sqr(pt_lists, xyd_cmp):
    """Find closest line that intersects and create a circle"""
    min_d = xyd_cmp[2]
    for i in range(pt_lists.shape[0]):
        (sqdist, success) = sqdist_ptline(pt_lists[i, 0], pt_lists[i, 1], xyd_cmp[:2])
        if success and sqdist < min_d:
            min_d = sqdist
    
    return min_d

def comb_dict(max_rays=MAX_RAYS, set_size=3, sideline_ct=4, seed=0):
    assert max_rays >= MIN_RAYS

    c_dict = {}
    for i in range(MIN_RAYS, max_rays + 1):
        combs = random_pairs(i + sideline_ct, set_size=set_size, seed=seed)
        ray_set = set(range(i))
        
        # Filter out impossible sets ie all sidelines or all rays
        c_dict[i] = [
            c for c in combs 
            if bool(ray_set.intersection(c)) and not
            set(c).issubset(ray_set)
            ]

    return c_dict

### Geometry Utilities
# @njit
def rot2d(th):
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th),  np.cos(th)]])

# @njit
def circle_line(xy1, xy2, sq_r, eps=0.001):
    """Intersection between circle and line to generate more circles
    Source: https://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    dx = xy2[0] - xy1[0]
    dy = xy2[1] - xy1[1]
    sq_dr = dx ** 2 + dy ** 2

    det = xy1[0] * xy2[1] - xy2[0] * xy1[1]
    sq_det = det **  2

    delta = sq_r * sq_dr - sq_det
    if delta < 0.0:
        return (None, False)

    sgn_dy = -1 if dy < 0 else 1
    sq_all = np.sqrt(sq_r * sq_dr - sq_det)
    x_1 = det * dy
    x_2 = sgn_dy * dx * sq_all

    y_1 = -det * dx
    y_2 = np.abs(dy) * sq_all
    
    x1 = (x_1 + x_2) / sq_dr
    y1 = (y_1 + y_2) / sq_dr
    if delta < eps:
        return (np.array([[x1, y1]]), True)
    
    x2 = (x_1 - x_2) / sq_dr
    y2 = (y_1 - y_2) / sq_dr
    
    return (np.array([[x1, y1], [x2, y2]]), True)

# @njit
def eqdist(xy1, xy2, xy3, eps=0.001, gamma=0.9):
    """Point along line that gives equal distance with point off line
    xy1 fixed point on line
    xy2 nonfixed point somewhere on line away from fixed xy1
    xy3 fixed point
    Note: distance between xy1 to xy2 and xy2 to xy3 are the same
    """
    x12 = xy1[0] + xy2[0]
    x23 = xy2[0] + xy3[0]
    y12 = xy1[1] + xy2[1]
    y23 = xy2[1] + xy3[1]

    det = x12 * x23 + y12 * y23 - x12 ** 2 - y12 ** 2
    if np.abs(det) < eps:
        return (None, False)

    u = (x23 ** 2 - x12 ** 2 + y23 ** 2 - y12 ** 2) / (2 * det)

    if u > 1 or u < 0:
        return (None, False)

    x = x12 * u * gamma - xy2[0]
    y = y12 * u * gamma - xy2[1]

    return (np.array([x, y]), True)

# @njit
def sqdist_ptline(xy1, xy2, xy3, eps=0.001): # x3,y3 is the point
    """Closest distance squared between a point and line segment
    x1, y1, x2, y2 define the line segment
    x3, y3 define the point
    """
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    px = xy2[0] - xy1[0]
    py = xy2[1] - xy1[1]

    norm = px * px + py * py
    if abs(norm) < eps:
        return (None, False)

    u =  ((xy3[0] - xy1[0]) * px + (xy3[1] - xy1[1]) * py) / norm

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = xy1[0] + u * px
    y = xy1[1] + u * py

    dx = x - xy3[0]
    dy = y - xy3[1]

    sqdist = dx * dx + dy * dy

    return (sqdist, True)

# @njit
def inter_rayline(pxy, rxy, qxy, sxy, eps=0.001, tmin=0, tmax=1, smin=0, smax=1):
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    s1_x = rxy[0] - pxy[0]
    s1_y = rxy[1] - pxy[1]
    s2_x = sxy[0] - qxy[0]
    s2_y = sxy[1] - qxy[1]

    det = -s2_x * s1_y + s1_x * s2_y

    # Parallel or colinear
    if np.abs(det) < eps:
        return (None, False)

    s = (-s1_y * (pxy[0] - qxy[0]) + s1_x * (pxy[1] - qxy[1])) / det
    t = ( s2_x * (pxy[1] - qxy[1]) - s2_y * (pxy[0] - qxy[0])) / det

    # Intersection with in bounds
    if tmin <= t and t <= tmax and smin <= s and s <= smax:
        i_x = pxy[0] + (t * s1_x)
        i_y = pxy[1] + (t * s1_y)
        return (np.array([i_x, i_y]), True)
    
    # Intersection but not in bounds
    return (None, False)

# @njit
def sqdist(xy1, xy2):
    "Calculates the distance sqaured between two points 2D"
    return (xy2[0] - xy1[0]) ** 2 + (xy2[1] - xy1[1]) ** 2

# @njit
def dist2pts(xy1, xy2):
    "Calculates the distance between two points 2D"
    return np.sqrt(sqdist(xy1, xy2))

# @njit
def tri_incenter(xy1, xy2, xy3, eps=0.001):
    """Calculate the coordinates of the incenter of the triangle and radius squared
    Note: For computational efficiency the squared of the radius is returned
    """
    a = dist2pts(xy2, xy3)
    b = dist2pts(xy3, xy1)
    c = dist2pts(xy1, xy2)

    # Poorly defined circle
    if a < eps or b < eps or c < eps:
        return (None, False)

    abc = a + b + c
    x = (a * xy1[0] + b * xy2[0] + c * xy3[0]) / abc
    y = (a * xy1[1] + b * xy2[1] + c * xy3[1]) / abc
    s = abc / 2
    sq_r = (s - a) * (s - b) * (s - c) / s
    return (np.array([x, y, sq_r]), True)

# TODO: Don't know why this doesn't work with njit
def cart2th(xy):
    return np.array([np.arctan2(xy[1], xy[0])])

def cart2pol(xy):
    rho = np.sqrt(xy[0] ** 2 + xy[1] ** 2)
    th = np.arctan2(xy[1], xy[0])
    return np.array([rho, th])

# @njit
def pol2cart(th, r):
    return np.array([r * np.cos(th), r * np.sin(th)])

# @njit
def wrap_pn_pi(x):
    """Wrap Pi from -Pi to +Pi"""
    return np.arctan2(np.sin(x), np.cos(x))

### Environment Utilities
def random_env(
        ball_xrange = (-FL                   ,  FL), 
        ball_yrange = (-(FW + field_extender), (FW + field_extender)), 
        team_xrange = (-FL                   ,  FL),
        team_yrange = (-(FW + field_extender), (FW + field_extender)),
        opp1_xrange = (-FL                   ,  FL),
        opp1_yrange = (-(FW + field_extender), (FW + field_extender)),
        opp2_xrange = (-FL                   ,  FL),
        opp2_yrange = (-(FW + field_extender), (FW + field_extender)),
        t1_eps = 0.5,
        ball_size = 0.2,
        robot_size = 0.4,
        seed=0,
        rng=None
        ):
    """Sample a random game state dictionary"""
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    b = np.array([[rng.uniform(ball_xrange[0], ball_xrange[1]),
        rng.uniform(ball_yrange[0], ball_yrange[1]), ball_size]])

    state_dict = {
        'b' : b,
        # "t1": np.array([[b[0, 0] - t1_eps, b[0, 1], 0]]),
        "t1": np.array([[
            rng.uniform(team_xrange[0], team_xrange[1]),
            rng.uniform(team_yrange[0], team_yrange[1]),
            robot_size            
        ]]),
        "t2": np.array([[
            rng.uniform(team_xrange[0], team_xrange[1]),
            rng.uniform(team_yrange[0], team_yrange[1]),
            robot_size
        ]]),
        "op1": np.array([[
            rng.uniform(opp1_xrange[0], opp1_xrange[1]),
            rng.uniform(opp1_yrange[0], opp1_yrange[1]),
            robot_size
        ]]),
        "op2": np.array([[
            rng.uniform(opp2_xrange[0], opp2_xrange[1]),
            rng.uniform(opp2_yrange[0], opp2_yrange[1]),
            robot_size
        ]]),
    }

    return (state_dict, rng)

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

### Functions taken from localization utilities/constants
# Note: Be careful as they may have been modified slightly
#       from original implementation
def random_pairs(num_items, set_size=2, seed=0):
    random.seed(seed)
    assert num_items > 1
    num_list = range(0, num_items)

    # Generate non-repeating pairs
    pairs = list(itertools.combinations(num_list, set_size))    

    # Shuffle
    random.shuffle(pairs)
    return pairs

### Pre-Computed Dictionaries
combinations_dict = comb_dict(max_rays=MAX_RAYS, set_size=3, sideline_ct=4, seed=0)

