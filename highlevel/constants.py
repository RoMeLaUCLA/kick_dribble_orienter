import numpy as np

# Field dimensions
field_length = 14
field_width = 9
circle_diameter = 3
goal_width = 2.6  
goal_depth = 0.6  # ADDED
goal_box_length = 1
goal_box_width = 4
penalty_cross_length = 2.1
penalty_box_length = 3
penalty_box_width = 6
green_edge = 1

max_length = field_length + green_edge
max_width = field_width + green_edge

# Labels from vision data
label2char = {}
label2char[0] = 'b'  # ball
label2char[1] = 'p'  # goal posts
label2char[2] = 'r'  # robot
label2char[3] = 'c'  # corner
label2char[4] = 't'  # T
label2char[5] = 'x'  # cross
label2char[6] = 's'  # sign: romela
char2label = dict([(value, key) for key, value in label2char.items()])

# Upper right quadrant of field with zero at center
# Cross pattern
center_x  = [char2label['x'], 0, 0]
circle_x  = [char2label['x'], 0, circle_diameter/2]
penalty_x = [char2label['x'], field_length/2 - penalty_cross_length, 0]

# T pattern
middle_T  = [char2label['t'], 0, field_width/2]
penalty_T = [char2label['t'], field_length/2, penalty_box_width/2]
goal_T    = [char2label['t'], field_length/2, goal_box_width/2]

# Corner pattern
field_corner   = [char2label['c'], field_length/2, field_width/2]
goal_corner    = [char2label['c'], field_length/2 - goal_box_length, goal_box_width/2]
penalty_corner = [char2label['c'], field_length/2 - penalty_box_length, penalty_box_width/2]

# Goal posts
post = [char2label['p'], field_length/2, goal_width/2]
post_behind = [char2label['p'], field_length/2 + goal_depth, goal_width/2]  # ADDED

# Generate symmetric points on field based on features
# Crosses
cross2 = np.array([circle_x, penalty_x])
cross1 = np.array([center_x])
cross_all = np.vstack((
    cross1, 
    cross2, 
    np.vstack((cross2[:, 0], -cross2[:, 1], -cross2[:, 2])).T
))

# Corners
corner4 = np.array([
    field_corner, 
    goal_corner, 
    penalty_corner
    ])
corner_all = np.vstack((
    corner4,
    np.vstack((corner4[:, 0], -corner4[:, 1], -corner4[:, 2])).T,
    np.vstack((corner4[:, 0], -corner4[:, 1],  corner4[:, 2])).T,
    np.vstack((corner4[:, 0],  corner4[:, 1], -corner4[:, 2])).T
))

# Ts
T4 = np.array([penalty_T, goal_T])
T2 = np.array([middle_T])
T_all = np.vstack((
    T2,
    np.vstack((T2[:, 0], -T2[:, 1], -T2[:, 2])).T,
    T4,
    np.vstack((T4[:, 0], -T4[:, 1], -T4[:, 2])).T,
    np.vstack((T4[:, 0], -T4[:, 1],  T4[:, 2])).T,
    np.vstack((T4[:, 0],  T4[:, 1], -T4[:, 2])).T
))

# Goal post
post4 = np.array([post, post_behind])  # ADDED
post_all = np.vstack((
    post4,
    np.vstack((post4[:, 0], -post4[:, 1], -post4[:, 2])).T,
    np.vstack((post4[:, 0], -post4[:, 1],  post4[:, 2])).T,
    np.vstack((post4[:, 0],  post4[:, 1], -post4[:, 2])).T
))

field_pts = np.vstack((cross_all, corner_all, T_all, post_all))