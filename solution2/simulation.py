# simulation.py
import numpy as np
from config import *

def get_missile_position(missile_id, t):
    """计算导弹在t时刻的位置 (无变化)"""
    p0 = INITIAL_POSITIONS_MISSILES[missile_id]
    dist_to_target = np.linalg.norm(p0)
    if dist_to_target < 1e-6: return p0
    direction = -p0 / dist_to_target
    position = p0 + direction * MISSILE_SPEED * t
    return position

def is_line_of_sight_blocked(p_missile, p_target, p_smoke_center):
    """判断单条视线是否被烟幕球体遮挡"""
    line_vec = p_target - p_missile
    dist_sq_missile_target = np.dot(line_vec, line_vec)
    
    if dist_sq_missile_target < 1e-6:
        return False # 距离太近，无法形成有效遮蔽

    smoke_to_missile_vec = p_smoke_center - p_missile
    
    # 投影长度的归一化值 (t_proj)
    t_proj = np.dot(smoke_to_missile_vec, line_vec) / dist_sq_missile_target
    
    # 投影点必须在导弹和目标之间
    if not (0 < t_proj < 1):
        return False
        
    # 垂直距离的平方
    dist_perp_sq = np.dot(smoke_to_missile_vec, smoke_to_missile_vec) - (t_proj**2) * dist_sq_missile_target

    # 垂直距离必须小于烟幕半径
    return dist_perp_sq < SMOKE_RADIUS**2

def calculate_total_obscuration_time(strategies, missile_id):
    """
    (已更新) 计算总遮蔽时间，使用48点精确模型
    """
    if not strategies:
        return 0.0

    obscuration_intervals = []
    for strategy in strategies:
        drone_id, v_drone, theta, t_drop, t_fuze = strategy
        
        p0_drone = INITIAL_POSITIONS_DRONES[drone_id]
        v_vec_drone = np.array([v_drone * np.cos(theta), v_drone * np.sin(theta), 0])
        p_drop = p0_drone + v_vec_drone * t_drop
        
        t_detonation = t_drop + t_fuze
        p_detonation = p_drop + v_vec_drone * t_fuze + np.array([0, 0, -0.5 * GRAVITY * t_fuze**2])
        
        sim_start_time = t_detonation
        sim_end_time = t_detonation + SMOKE_EFFECTIVE_DURATION
        
        current_interval_start = -1

        for t in np.arange(sim_start_time, sim_end_time, SIMULATION_TIME_STEP):
            p_missile = get_missile_position(missile_id, t)
            p_smoke_center = p_detonation + np.array([0, 0, -SMOKE_SINK_SPEED * (t - t_detonation)])
            
            # --- 核心修改：检查所有48个顶点 ---
            is_fully_obscured_this_step = True
            for p_vertex in TARGET_VERTICES:
                if not is_line_of_sight_blocked(p_missile, p_vertex, p_smoke_center):
                    is_fully_obscured_this_step = False
                    break # 优化：一旦有一个点可见，则遮蔽失败，无需再检查
            
            # 更新遮蔽区间
            if is_fully_obscured_this_step and current_interval_start < 0:
                current_interval_start = t
            elif not is_fully_obscured_this_step and current_interval_start >= 0:
                obscuration_intervals.append((current_interval_start, t))
                current_interval_start = -1
        
        if current_interval_start >= 0:
            obscuration_intervals.append((current_interval_start, sim_end_time))
    
    # --- 合并时间区间 (无变化) ---
    if not obscuration_intervals: return 0.0
    obscuration_intervals.sort()
    merged = [obscuration_intervals[0]]
    for current_start, current_end in obscuration_intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start < last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    total_time = sum(end - start for start, end in merged)
    return total_time