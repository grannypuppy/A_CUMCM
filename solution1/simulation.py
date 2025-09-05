# simulation.py
import numpy as np
from config import *

def get_missile_position(missile_id, t):
    """计算导弹在t时刻的位置"""
    p0 = INITIAL_POSITIONS_MISSILES[missile_id]
    dist_to_target = np.linalg.norm(p0)
    direction = -p0 / dist_to_target
    position = p0 + direction * MISSILE_SPEED * t
    return position

def calculate_total_obscuration_time(strategies, missile_id):
    """
    计算一组策略对单个导弹的总遮蔽时间。
    strategies: list of tuples, 每个tuple是一个策略 (drone_id, v, theta, t_drop, t_fuze)
    """
    if not strategies:
        return 0.0

    # 计算每个烟幕云的有效遮蔽时间区间
    obscuration_intervals = []
    for strategy in strategies:
        drone_id, v_drone, theta, t_drop, t_fuze = strategy
        
        # 1. 计算无人机和烟幕弹轨迹
        p0_drone = INITIAL_POSITIONS_DRONES[drone_id]
        v_vec_drone = np.array([v_drone * np.cos(theta), v_drone * np.sin(theta), 0])
        p_drop = p0_drone + v_vec_drone * t_drop
        
        t_detonation = t_drop + t_fuze
        # 抛物线位移 s = v0*t + 0.5*a*t^2
        p_detonation = p_drop + v_vec_drone * t_fuze + np.array([0, 0, -0.5 * GRAVITY * t_fuze**2])
        
        # 2. 模拟遮蔽过程
        start_time = -1
        is_obscuring = False
        
        sim_start_time = t_detonation
        sim_end_time = t_detonation + SMOKE_EFFECTIVE_DURATION
        
        for t in np.arange(sim_start_time, sim_end_time, SIMULATION_TIME_STEP):
            p_missile = get_missile_position(missile_id, t)
            
            # 烟幕中心下沉
            p_smoke_center = p_detonation + np.array([0, 0, -SMOKE_SINK_SPEED * (t - t_detonation)])
            
            # 简化版：只检查到目标中心轴中点的视线
            p_target_mid = (TARGET_AXIS_BOTTOM + TARGET_AXIS_TOP) / 2.0
            
            # 几何判断
            line_vec = p_target_mid - p_missile
            dist_sq_missile_target = np.dot(line_vec, line_vec)
            
            if dist_sq_missile_target > 1e-6:
                smoke_to_missile_vec = p_smoke_center - p_missile
                # 投影长度
                proj_len = np.dot(smoke_to_missile_vec, line_vec) / dist_sq_missile_target
                
                # 垂直距离的平方
                dist_perp_sq = np.dot(smoke_to_missile_vec, smoke_to_missile_vec) - proj_len**2 * dist_sq_missile_target

                if 0 < proj_len < 1 and dist_perp_sq < SMOKE_RADIUS**2:
                    if not is_obscuring:
                        is_obscuring = True
                        start_time = t
                else:
                    if is_obscuring:
                        is_obscuring = False
                        obscuration_intervals.append((start_time, t))
            else: # 导弹已到达目标
                 if is_obscuring:
                    is_obscuring = False
                    obscuration_intervals.append((start_time, t))

        if is_obscuring: # 处理模拟结束时仍在遮蔽的情况
            obscuration_intervals.append((start_time, sim_end_time))

    # 合并所有重叠的时间区间并计算总时长
    if not obscuration_intervals:
        return 0.0
    
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