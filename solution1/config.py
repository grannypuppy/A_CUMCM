# config.py
import numpy as np

# ========== 坐标系与关键目标 ==========
# 真目标：圆柱体
TARGET_CENTER_BASE = np.array([0, 200, 0])  # 底面圆心
TARGET_RADIUS = 7  # m
TARGET_HEIGHT = 10  # m
TARGET_PROTECTED_POINTS_COUNT = 24  # 上下圆周的采样点数

# ========== 运动学模型 ==========
# 导弹
V_MISSILE = 300  # m/s

# 无人机
V_DRONE_MIN = 70  # m/s
V_DRONE_MAX = 140  # m/s
FUSE_TIME_MIN = 1 # s, 假设最小引信时间
FUSE_TIME_MAX = 10 # s, 假设最大引信时间

# 烟幕弹/云团
G = 9.8  # m/s^2
V_SINK = 3  # m/s
SMOKE_RADIUS = 10  # m
SMOKE_DURATION = 20  # s

# ========== 模拟与优化参数 ==========
# 时间离散化
T_START = 0
T_END = 60 # 模拟总时长，s
DT = 5    # 时间步长，s

# 场景设置
# 导弹初始位置 (x, y, z)
MISSILE_INITIAL_POSITIONS = [
    np.array([20000, 15000, 5000]),
    np.array([-18000, 16000, 4500])
]

# 无人机初始位置
DRONE_INITIAL_POSITIONS = [
    np.array([500, 500, 5000]),
    np.array([-500, -500, 5000]),
    np.array([1000, -1000, 5000])
]

# 每架无人机的烟幕弹数量
DRONES_BOMBS = [2, 2, 2]

# Gurobi 参数
GUROBI_TIME_LIMIT = 300 # Gurobi 求解时间限制, s
GUROBI_MIP_GAP = 0.1 # Gurobi MIP Gap