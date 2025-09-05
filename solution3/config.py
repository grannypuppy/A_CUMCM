# config.py
import numpy as np

# 物理常量
GRAVITY = 9.8  # m/s^2

# 导弹参数
MISSILE_SPEED = 300.0  # m/s
INITIAL_POSITIONS_MISSILES = {
    "M1": np.array([20000, 0, 2000]), #
    "M2": np.array([19000, 600, 2100]), #
    "M3": np.array([18000, -600, 1900]), #
}

# 无人机参数
DRONE_SPEED_RANGE = (70.0, 140.0)  # m/s
DECOYS_PER_DRONE = 3 # 固定为3
MIN_INTERVAL_DECOYS = 1.0 # s
INITIAL_POSITIONS_DRONES = {
    "FY1": np.array([17800, 0, 1800]), #
    "FY2": np.array([12000, 1400, 1400]), #
    "FY3": np.array([6000, -3000, 700]), #
    "FY4": np.array([11000, 2000, 1800]), #
    "FY5": np.array([13000, -2000, 1300]), #
}

# 烟幕云团参数
SMOKE_RADIUS = 10.0  # m
SMOKE_SINK_SPEED = 3.0  # m/s
SMOKE_EFFECTIVE_DURATION = 20.0  # s

# 目标参数
TARGET_CENTER = np.array([0, 200, 0]) #
TARGET_RADIUS = 7.0 # m
TARGET_HEIGHT = 10.0 # m

# --- 预计算目标顶点 ---
NUM_VERTICES_PER_CIRCLE = 24
TARGET_VERTICES = []
# 上下两个圆面
for z in [0, TARGET_HEIGHT]:
    for i in range(NUM_VERTICES_PER_CIRCLE):
        angle = 2 * np.pi * i / NUM_VERTICES_PER_CIRCLE
        # 使用外切多边形，半径需要除以cos(pi/N)
        # 为简化，此处用内接多边形顶点。如需更严格，可替换为外切
        r = TARGET_RADIUS 
        x = TARGET_CENTER[0] + r * np.cos(angle)
        y = TARGET_CENTER[1] + r * np.sin(angle)
        TARGET_VERTICES.append(np.array([x, y, z]))
TARGET_VERTICES = np.array(TARGET_VERTICES) # 转换为numpy数组，共48个点

# 模拟参数
SIMULATION_MAX_TIME = 80 # s, 估算一个最长模拟时间
SIMULATION_TIME_STEP = 0.2 # s, 适当调大步长以加快计算