# config.py
import numpy as np

# --- 物理常量和基础参数 (与之前相同) ---
GRAVITY = 9.8
MISSILE_SPEED = 300.0
DRONE_SPEED_RANGE = (70.0, 140.0)
MAX_DECOYS_PER_DRONE = 3
MIN_INTERVAL_DECOYS = 1.0
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_EFFECTIVE_DURATION = 20.0
SIMULATION_TIME_STEP = 0.1 # s, 在高精度模型下可适当增大此值以提速, e.g., 0.2s

# --- 初始位置 (与之前相同) ---
INITIAL_POSITIONS_MISSILES = {
    "M1": np.array([20000, 0, 2000]),
    "M2": np.array([19000, 600, 2100]),
    "M3": np.array([18000, -600, 1900]),
}
INITIAL_POSITIONS_DRONES = {
    "FY1": np.array([17800, 0, 1800]),
    "FY2": np.array([12000, 1400, 1400]),
    "FY3": np.array([6000, -3000, 700]),
    "FY4": np.array([11000, 2000, 1800]),
    "FY5": np.array([13000, -2000, 1300]),
}

# --- 目标参数 (已更新以包含48个顶点) ---
TARGET_CENTER = np.array([0, 200, 0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0
NUM_VERTICES_PER_CIRCLE = 24 # 您提出的24边形拟合

def generate_target_vertices():
    """生成拟合圆柱体上下圆周的48个顶点"""
    vertices = []
    angles = np.linspace(0, 2 * np.pi, NUM_VERTICES_PER_CIRCLE, endpoint=False)
    
    # 底面圆周上的点
    for angle in angles:
        x = TARGET_CENTER[0] + TARGET_RADIUS * np.cos(angle)
        y = TARGET_CENTER[1] + TARGET_RADIUS * np.sin(angle)
        vertices.append(np.array([x, y, 0]))
        
    # 顶面圆周上的点
    for angle in angles:
        x = TARGET_CENTER[0] + TARGET_RADIUS * np.cos(angle)
        y = TARGET_CENTER[1] + TARGET_RADIUS * np.sin(angle)
        vertices.append(np.array([x, y, TARGET_HEIGHT]))
        
    return np.array(vertices)

# 预计算目标顶点
TARGET_VERTICES = generate_target_vertices()