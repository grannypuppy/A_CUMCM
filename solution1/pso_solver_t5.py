import numpy as np
from simulator import solve_smoke_deployment
import time

# --- 1. 问题定义 ---
# 这些参数与 ppo_solver.py 中的定义相匹配
problem_missiles = {
        'M1': (20000, 0, 2000),
        'M2': (19000, 600, 2100),
        'M3': (18000, -600, 1900),
    }
problem_drones = {
        'FY1': (17800, 0, 1800),
        'FY2': (12000, 1400, 1400),
        'FY3': (6000, -3000, 700),
        'FY4': (11000, 2000, 1800),
        'FY5': (13000, -2000, 1300),
    }
problem_num_bombs = 3
problem_time_horizon = 80
problem_time_step = 0.005
MIN_DROP_INTERVAL = 1.0

# --- 开关：是否使用专家解作为初始粒子 ---
USE_EXPERT_SOLUTION = True # 设置为 False 则不使用


# --- 专家初始解 ---
# 该解来源于 simulator.py 中的示例
expert_flight_info = {
    'FY1': (139.99, np.pi * 179.65 / 180),
    'FY2': (139.71, np.pi * 256.72 / 180),
    'FY3': (124.67, np.pi * 74.86 / 180),
    'FY4': (101.90, np.pi * 245.64 / 180),
    'FY5': (135.84, np.pi * 118.38 / 180),
}
expert_bombs_info = {
    ('FY1', 0): (0.00, 3.61),
    ('FY1', 1): (3.65, 5.33),
    ('FY1', 2): (5.55, 6.05),
    ('FY2', 0): (2.51, 4.61),
    ('FY2', 1): (3.71, 3.64),
    ('FY2', 2): (6.47, 11.36),
    ('FY3', 0): (8.52, 5.55),
    ('FY3', 1): (24.23, 1.41),
    ('FY3', 2): (25.26, 0.27),
    ('FY4', 0): (8.41, 12.63),
    ('FY4', 1): (55.04, 8.74),
    ('FY4', 2): (39.59, 4.02),
    ('FY5', 0): (11.84, 1.67),
    ('FY5', 1): (1.26, 0.62),
    ('FY5', 2): (19.92, 0.00),
}

def encode_solution_to_particle(flight_info, bombs_info, drones_info, num_bombs_per_drone):
    """将一个已知的决策方案编码为粒子向量。"""
    particle = []
    # 对无人机ID进行排序以确保编码和解码的顺序一致
    drone_ids = sorted(list(drones_info.keys()))
    
    # 1. 编码无人机参数
    for drone_id in drone_ids:
        speed, angle = flight_info[drone_id]
        particle.extend([speed, angle])

    # 2. 编码烟雾弹参数
    for drone_id in drone_ids:
        for k in range(num_bombs_per_drone):
            t_drop, fusetime = bombs_info[(drone_id, k)]
            particle.extend([t_drop, fusetime])
            
    return np.array(particle)


def fitness_function(particle, drones_info, num_bombs_per_drone, time_horizon):
    """
    评估单个粒子的适应度。

    Args:
        particle (np.array): 代表一个解决方案的粒子位置向量。
        drones_info (dict): 无人机信息。
        num_bombs_per_drone (int): 每架无人机的烟雾弹数量。
        time_horizon (int): 仿真时间。

    Returns:
        float: 该粒子的适应度分数 (总遮蔽时间 - 惩罚)。
    """
    flight_info = {}
    bombs_info = {}
    particle_idx = 0

    drone_ids = sorted(list(drones_info.keys()))
    
    # 1. 解码无人机参数
    for drone_id in drone_ids:
        speed = particle[particle_idx]
        angle = particle[particle_idx + 1]
        flight_info[drone_id] = (speed, angle)
        particle_idx += 2

    # 2. 解码烟雾弹参数
    drop_times_for_drones = {drone_id: [] for drone_id in drone_ids}
    for drone_id in drone_ids:
        for k in range(num_bombs_per_drone):
            t_drop = particle[particle_idx]
            fusetime = particle[particle_idx + 1]
            bombs_info[(drone_id, k)] = (t_drop, fusetime)
            drop_times_for_drones[drone_id].append(t_drop)
            particle_idx += 2

    # 3. 计算对违反最小投掷间隔的惩罚
    penalty = 0
    for drone_id in drone_ids:
        sorted_drop_times = sorted(drop_times_for_drones[drone_id])
        for i in range(len(sorted_drop_times) - 1):
            if sorted_drop_times[i+1] - sorted_drop_times[i] < MIN_DROP_INTERVAL:
                penalty += 50 * (MIN_DROP_INTERVAL - (sorted_drop_times[i+1] - sorted_drop_times[i]))

    # 4. 运行模拟
    total_coverage = solve_smoke_deployment(
        missiles_info=problem_missiles,
        drones_info=drones_info,
        flight_info=flight_info,
        bombs_info=bombs_info,
        num_bombs_per_drone=num_bombs_per_drone,
        time_horizon=time_horizon,
        time_step=problem_time_step,
        verbose=False 
    )

    return total_coverage - penalty

def pso_solver(drones_info, num_bombs_per_drone, time_horizon):
    """
    使用粒子群优化算法求解烟幕屏部署问题。
    """
    # --- PSO 参数 ---
    num_particles = 400
    max_generations = 1000
    w = 0.5  # 惯性权重
    c1 = 1.5 # 认知系数
    c2 = 1.5 # 社会系数

    num_drones = len(drones_info)
    dim = (num_drones * 2) + (num_drones * num_bombs_per_drone * 2)

    # --- 变量边界 ---
    # [speed, angle, t_drop_1, f_time_1, t_drop_2, f_time_2, ...]
    lb = []
    ub = []
    for _ in range(num_drones):
        lb.extend([70, 0]) # speed, angle
        ub.extend([140, 2 * np.pi])
        for _ in range(num_bombs_per_drone):
            lb.extend([0, 1]) # t_drop, fusetime
            ub.extend([time_horizon / 2, 10])
    lb = np.array(lb)
    ub = np.array(ub)

    # --- 初始化 ---
    particles_pos = lb + (ub - lb) * np.random.rand(num_particles, dim)
    
    # --- 注入专家解 (如果开关为True) ---
    if USE_EXPERT_SOLUTION:
        # 将第一个粒子替换为已知的优秀解，以引导优化过程
        expert_particle = encode_solution_to_particle(
            expert_flight_info, expert_bombs_info, drones_info, num_bombs_per_drone
        )
        particles_pos[0] = expert_particle
        print("已将专家解注入粒子群。")
    
    particles_vel = np.zeros((num_particles, dim))
    
    pbest_pos = particles_pos.copy()
    pbest_fitness = np.array([fitness_function(p, drones_info, num_bombs_per_drone, time_horizon) for p in particles_pos])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    print(f"开始PSO优化... 维度: {dim}, 粒子数: {num_particles}, 最大代数: {max_generations}")
    start_time = time.time()

    # --- PSO 主循环 ---
    for gen in range(max_generations):
        for i in range(num_particles):
            # 更新速度
            r1, r2 = np.random.rand(2)
            cognitive_vel = c1 * r1 * (pbest_pos[i] - particles_pos[i])
            social_vel = c2 * r2 * (gbest_pos - particles_pos[i])
            particles_vel[i] = w * particles_vel[i] + cognitive_vel + social_vel

            # 更新位置
            particles_pos[i] = particles_pos[i] + particles_vel[i]

            # 边界处理
            particles_pos[i] = np.maximum(particles_pos[i], lb)
            particles_pos[i] = np.minimum(particles_pos[i], ub)

            # 评估适应度
            current_fitness = fitness_function(particles_pos[i], drones_info, num_bombs_per_drone, time_horizon)

            # 更新 pbest
            if current_fitness > pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = particles_pos[i].copy()

                # 更新 gbest
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = particles_pos[i].copy()
        
        if (gen + 1) % 10 == 0:
            print(f"Generation [{gen+1}/{max_generations}], Best Fitness: {gbest_fitness:.4f}")

    end_time = time.time()
    print(f"PSO优化完成！耗时: {end_time - start_time:.2f} 秒")
    
    return gbest_pos, gbest_fitness

def print_solution(best_particle, drones_info, num_bombs_per_drone):
    """打印格式化的最优解"""
    flight_info = {}
    bombs_info = {}
    particle_idx = 0
    drone_ids = sorted(list(drones_info.keys()))

    for drone_id in drone_ids:
        flight_info[drone_id] = (best_particle[particle_idx], best_particle[particle_idx + 1])
        particle_idx += 2
    for drone_id in drone_ids:
        for k in range(num_bombs_per_drone):
            bombs_info[(drone_id, k)] = (best_particle[particle_idx], best_particle[particle_idx + 1])
            particle_idx += 2
            
    print("\n--- PSO找到的最优策略 ---")
    for j, (speed, angle) in flight_info.items():
        print(f"无人机 {j}:")
        print(f"  飞行速度: {speed:.2f} m/s")
        print(f"  飞行角度: {np.rad2deg(angle):.2f} 度")
    
    sorted_bombs = sorted(bombs_info.items(), key=lambda item: (item[0][0], item[1][0]))
    for (j, k), (t_drop, fusetime) in sorted_bombs:
        print(f"烟雾弹 ({j}, {k}):")
        print(f"  投掷时间: {t_drop:.2f} s")
        print(f"  引信时间: {fusetime:.2f} s")
        print(f"  -> 引爆时间: {t_drop + fusetime:.2f} s")


if __name__ == '__main__':
    best_solution_particle, best_fitness = pso_solver(
        drones_info=problem_drones,
        num_bombs_per_drone=problem_num_bombs,
        time_horizon=problem_time_horizon
    )
    
    print("\n=============================================")
    print(f"最优适应度 (总遮蔽时间): {best_fitness:.2f} 秒")
    print("=============================================")
    
    print_solution(best_solution_particle, problem_drones, problem_num_bombs)

    print("\n--- 使用最优策略进行最终高精度模拟评估 ---")
    # 使用最优解运行一次高精度模拟
    final_flight_info = {}
    final_bombs_info = {}
    particle_idx = 0
    drone_ids = sorted(list(problem_drones.keys()))
    for drone_id in drone_ids:
        final_flight_info[drone_id] = (best_solution_particle[particle_idx], best_solution_particle[particle_idx + 1])
        particle_idx += 2
    for drone_id in drone_ids:
        for k in range(problem_num_bombs):
            final_bombs_info[(drone_id, k)] = (best_solution_particle[particle_idx], best_solution_particle[particle_idx + 1])
            particle_idx += 2
            
    final_coverage = solve_smoke_deployment(
        missiles_info=problem_missiles,
        drones_info=problem_drones,
        flight_info=final_flight_info,
        bombs_info=final_bombs_info,
        num_bombs_per_drone=problem_num_bombs,
        time_horizon=problem_time_horizon,
        time_step=0.01, # 使用更高的时间精度
        verbose=True
    )
    print("\n=============================================")
    print(f"最终高精度评估 - 总有效遮蔽时间: {final_coverage:.2f} 秒")
    print("=============================================")
