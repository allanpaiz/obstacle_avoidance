"""The one-off TurtleBot4 obstacle-avoidance run, staged in the demo arena.

The arena is the heterogeneous-robot demo's (4.8 m x 3.4 m, centre island,
route loop) with only turtlebot4_1 in it. turtlebot4_1 drives the north leg of
the route westward; a safety barrel stands in the middle of it, a little left
of the line, so the detour goes right, between the island and the wall. The
overview camera looks from the north, so the robot passes in front of the
barrel rather than behind it.

Frames: the controller plans in the robot's start frame (x forward, y left),
the same frame the hardware tb4_sensor_detection pipeline plans in.
"""

from __future__ import annotations

import math


WORLD_NAME = "heterogeneous_robot_demo"           # world.py
CLOCK_TOPIC = f"/world/{WORLD_NAME}/clock"
POSE_TOPIC = f"/world/{WORLD_NAME}/dynamic_pose/info"

ROBOT = "turtlebot4_1"
ROBOT_COLOR = "0.06 0.48 0.82 1"
START = (1.9, 0.9, math.pi)                       # world x, y, yaw

BARREL_NAME = "obstacle_barrel"
BARREL_XY = (0.0, 0.84)
BARREL_RADIUS = 0.15
BARREL_HEIGHT = 0.50

# Detection: the hardware detector's defaults (detect_plan_execute_straight.py).
HOLD_S = 1.0                                      # stand still before detecting
DETECT_S = 2.0                                    # scans are collected for this long
MAX_RANGE = 3.0
FRONT_FOV_DEG = 160.0
CLUSTER_EPS = 0.12
MIN_SAMPLES = 4
RADIUS_PADDING = 0.04
CORRIDOR_WIDTH = 0.7
PLANNER_RADIUS_MIN = 0.20
MIN_X = 0.25                                      # ignores returns from the robot's own tower
MAX_OBSTACLE_RADIUS = 0.6                         # walls are not obstacles to plan around

# Planning and tracking.
GOAL_DISTANCE = 3.4                               # to the route corner at (-1.5, 0.9)
ROBOT_RADIUS = 0.17
SAFETY_MARGIN = 0.08
RAMP_LENGTH = 0.8                                 # metres to swing out, and to swing back
CRUISE_SPEED = 0.22
LOOKAHEAD = 0.30
GOAL_TOLERANCE = 0.04
CONTROL_RATE = 20.0

# Sensors. The lidar matches the TurtleBot4 RPLIDAR geometry
# (640 rays, 0.164-12 m); every rate is a divisor of the lidar's.
CAMERA_TOPIC = f"/{ROBOT}/camera/image"
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_RATE, CAMERA_HFOV = 640, 480, 10.0, 1.25
CAMERA_POSE = "0.19 0 0.20 0 0.10 0"              # front of the base, clear of the tower
LIDAR_TOPIC = f"/model/{ROBOT}/scan"
LIDAR_SAMPLES, LIDAR_RATE, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE = 640, 20.0, 0.164, 12.0
LIDAR_POSE = "-0.04 0 0.192915 0 0 0"             # on the rplidar mesh
OVERVIEW_TOPIC = "/demo/overview/image"
OVERVIEW_WIDTH, OVERVIEW_HEIGHT, OVERVIEW_RATE, OVERVIEW_HFOV = 960, 540, 10.0, 0.9
_PITCH, _DISTANCE, _AIM_Y = 0.85, 5.4, 0.3        # 3/4 view from the north
OVERVIEW_POSE = (f"0 {_AIM_Y + _DISTANCE * math.cos(_PITCH):.3f} {_DISTANCE * math.sin(_PITCH):.3f} "
                 f"0 {_PITCH} -1.5708")
ODOM_TOPIC = f"/model/{ROBOT}/odometry"
CMD_TOPIC = f"/model/{ROBOT}/cmd_vel"

TAIL_S = 2.5                                      # simulation seconds recorded after the goal
MAX_DURATION_S = 60.0
