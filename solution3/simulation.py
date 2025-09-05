# simulation.py
import numpy as np
from config import *

def get_missile_position(missile_id, t):
    """计算导弹在t时刻的位置"""
    p0 = INITIAL_POSITIONS_MISSILES[missile_id]
    dist_to_target = np.linalg.norm(p0)
    if dist_to_target == 0: return p0
    direction = -p0 / dist_to_target
    position = p0 + direction * MISSILE_SPEED * t
    return position

def calculate_fitness(chromosome):
    """
    全新的适应度函数，严格按照最终模型计算总遮蔽时间。
    chromosome: 长度为40的基因列表
    """
    # 1. 解码染色体并生成15个烟幕弹的完整生命周期信息
    decoys_info = []
    drone_ids = list(INITIAL_POSITIONS_DRONES.keys())
    
    for i in range(len(drone_ids)):
        drone_idx_start = i * 8
        genes = chromosome[drone_idx_start : drone_idx_start + 8]
        
        speed, angle = genes[0], genes[1]
        drop_times = sorted([genes[2], genes[4], genes[6]])
        
        # 约束检查：投放间隔
        if drop_times[1] - drop_times[0] < MIN_INTERVAL_DECOYS or \
           drop_times[2] - drop_times[1] < MIN_INTERVAL_DECOYS:
            return 0.0 # 无效解，适应度为0

        p0_drone = INITIAL_POSITIONS_DRONES[drone_ids[i]]
        v_vec_drone = np.array([speed * np.cos(angle), speed * np.sin(angle), 0])
        
        # 找到基因对应的引信时间
        t_fuzes = {}
        t_fuzes[genes[2]] = genes[3]
        t_fuzes[genes[4]] = genes[5]
        t_fuzes[genes[6]] = genes[7]
        
        for t_drop in drop_times:
            t_fuze = t_fuzes[t_drop]
            t_detonation = t_drop + t_fuze
            p_drop = p0_drone + v_vec_drone * t_drop
            p_detonation = p_drop + v_vec_drone * t_fuze + np.array([0, 0, -0.5 * GRAVITY * t_fuze**2])
            decoys_info.append({
                "detonation_pos": p_detonation,
                "start_time": t_detonation,
                "end_time": t_detonation + SMOKE_EFFECTIVE_DURATION
            })

    # 2. 全局遮蔽模拟
    total_obscuration_time = 0
    missile_ids = list(INITIAL_POSITIONS_MISSILES.keys())

    for t in np.arange(0, SIMULATION_MAX_TIME, SIMULATION_TIME_STEP):
        # 获取当前时刻所有有效烟幕云的位置
        active_smoke_centers = []
        for decoy in decoys_info:
            if decoy["start_time"] <= t <= decoy["end_time"]:
                dt_sink = t - decoy["start_time"]
                center = decoy["detonation_pos"] + np.array([0, 0, -SMOKE_SINK_SPEED * dt_sink])
                active_smoke_centers.append(center)
        
        if not active_smoke_centers:
            continue

        # 检查t时刻是否完全遮蔽
        is_fully_obscured_at_t = True
        for missile_id in missile_ids:
            p_missile = get_missile_position(missile_id, t)
            
            for p_vertex in TARGET_VERTICES:
                is_vertex_safe = False
                line_of_sight_vec = p_vertex - p_missile
                
                for p_smoke in active_smoke_centers:
                    # 快速几何判断
                    smoke_to_missile_vec = p_smoke - p_missile
                    proj_ratio = np.dot(smoke_to_missile_vec, line_of_sight_vec) / np.dot(line_of_sight_vec, line_of_sight_vec)
                    
                    if 0 < proj_ratio < 1:
                        dist_perp_sq = np.dot(smoke_to_missile_vec, smoke_to_missile_vec) - (proj_ratio**2) * np.dot(line_of_sight_vec, line_of_sight_vec)
                        if dist_perp_sq < SMOKE_RADIUS**2:
                            is_vertex_safe = True
                            break # 此顶点安全，检查下一个顶点
                
                if not is_vertex_safe:
                    is_fully_obscured_at_t = False
                    break # 此顶点不安全，则t时刻不满足完全遮蔽
            
            if not is_fully_obscured_at_t:
                break # 此导弹能看到目标，则t时刻不满足完全遮蔽

        if is_fully_obscured_at_t:
            total_obscuration_time += SIMULATION_TIME_STEP
            
    return total_obscuration_time