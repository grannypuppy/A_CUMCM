# config.py
# 存储所有问题参数和初始条件

import numpy as np

# 物理常量
GRAVITY = 9.8  # m/s^2

# 导弹参数 [源: 6]
MISSILE_SPEED = 300.0  # m/s [源: 6]
INITIAL_POSITIONS_MISSILES = {
    "M1": np.array([20000, 0, 2000]),
    "M2": np.array([19000, 600, 2100]),
    "M3": np.array([18000, -600, 1900]),
}

# 无人机参数 [源: 7]
DRONE_SPEED_RANGE = (70.0, 140.0)  # m/s [源: 7]
MAX_DECOYS_PER_DRONE = 3
MIN_INTERVAL_DECOYS = 1.0 # s [源: 5]
INITIAL_POSITIONS_DRONES = {
    "FY1": np.array([17800, 0, 1800]),
    "FY2": np.array([12000, 1400, 1400]),
    "FY3": np.array([6000, -3000, 700]),
    "FY4": np.array([11000, 2000, 1800]),
    "FY5": np.array([13000, -2000, 1300]),
}

# 烟幕云团参数
SMOKE_RADIUS = 10.0  # m [源: 5]
SMOKE_SINK_SPEED = 3.0  # m/s [源: 5]
SMOKE_EFFECTIVE_DURATION = 20.0  # s [源: 5]

# 目标参数
TARGET_POSITION = np.array([0, 200, 0]) # 圆心 [源: 6]
TARGET_RADIUS = 7.0 # m [源: 6]
TARGET_HEIGHT = 10.0 # m [源: 6]
# 简化为中心轴线用于视线判断
TARGET_AXIS_BOTTOM = np.array([0, 200, 0])
TARGET_AXIS_TOP = np.array([0, 200, 10])

# 模拟参数
SIMULATION_TIME_STEP = 0.1 # s