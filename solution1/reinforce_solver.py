"""
使用 REINFORCE (策略梯度) 算法解决烟幕屏部署问题的求解器。

该脚本需要安装以下库:
pip install torch gymnasium numpy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time

# --- 1. 从 ppo_solver.py 中借鉴的核心模拟逻辑和常量 ---

# 物理和场景常量
V_MISSILE = 300.0  # m/s
V_SINK = 3.0  # m/s
R_SMOKE = 10.0  # m
G = 9.8  # m/s^2
MIN_DROP_INTERVAL = 1.0 # s
SMOKE_DURATION = 20.0 # s
TARGET_CENTER_BASE = np.array([0, 200, 0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0

def generate_target_points():
    """生成目标圆柱体上的48个离散点"""
    target_points_list = []
    for h in [0, TARGET_HEIGHT]:
        for i in range(24):
            angle = 2 * np.pi * i / 24
            x = TARGET_CENTER_BASE[0] + TARGET_RADIUS * np.cos(angle)
            y = TARGET_CENTER_BASE[1] + TARGET_RADIUS * np.sin(angle)
            z = TARGET_CENTER_BASE[2] + h
            target_points_list.append(np.array([x, y, z]))
    return target_points_list

# 全局变量，避免重复计算
TARGET_POINTS = generate_target_points()

def decode_action(action, drones_info, num_bombs_per_drone, time_horizon):
    """将归一化的动作向量 [-1, 1] 映射到真实的物理值范围"""
    bombs = [(j, k) for j in drones_info.keys() for k in range(num_bombs_per_drone)]
    
    flight_info = {}
    bombs_info = {}
    
    action_idx = 0
    
    # 1. 解码无人机飞行参数
    for drone_id in drones_info.keys():
        speed = 105 + action[action_idx] * 35
        action_idx += 1
        angle = np.pi + action[action_idx] * np.pi
        action_idx += 1
        flight_info[drone_id] = (speed, angle)

    # 2. 解码每个烟雾弹的部署参数
    for bomb_id in bombs:
        t_drop = (time_horizon / 4) + action[action_idx] * (time_horizon / 4)
        action_idx += 1
        fusetime = 5.5 + action[action_idx] * 4.5
        action_idx += 1
        bombs_info[bomb_id] = (t_drop, fusetime)
        
    return flight_info, bombs_info

def encode_solution_to_action(flight_info, bombs_info, drones_info, num_bombs_per_drone, time_horizon):
    """
    将一个已知的决策方案 (物理单位) 编码为归一化的动作向量 [-1, 1].
    这是 decode_action 方法的逆操作。
    """
    bombs = [(j, k) for j in drones_info.keys() for k in range(num_bombs_per_drone)]
    action = []

    # 1. 编码无人机飞行参数
    for drone_id in drones_info.keys():
        speed, angle = flight_info[drone_id]
        # 速度: [70, 140] -> [-1, 1]
        norm_speed = (speed - 105) / 35
        action.append(norm_speed)
        # 角度: [0, 2*pi] -> [-1, 1]
        norm_angle = (angle - np.pi) / np.pi
        action.append(norm_angle)

    # 2. 编码烟雾弹部署参数
    for bomb_id in bombs:
        t_drop, fusetime = bombs_info[bomb_id]
        # 投掷时间: [0, time_horizon/2] -> [-1, 1]
        norm_t_drop = (t_drop - (time_horizon / 4)) / (time_horizon / 4)
        action.append(norm_t_drop)
        # 引信时间: [1, 10] -> [-1, 1]
        norm_fusetime = (fusetime - 5.5) / 4.5
        action.append(norm_fusetime)
        
    return np.array(action, dtype=np.float32)

def run_simulation(flight_info, bombs_info, missiles_info, drones_info, num_bombs_per_drone, time_horizon, time_step, verbose=False):
    """根据给定的决策执行一次完整的物理模拟并返回奖励"""
    
    bombs = [(j, k) for j in drones_info.keys() for k in range(num_bombs_per_drone)]

    # --- 1. 初始化决策变量 ---
    drone_speed, drone_angle = {}, {}
    drone_cos_angle, drone_sin_angle = {}, {}
    for j, (sp, theta) in flight_info.items():
        drone_speed[j] = sp
        drone_angle[j] = theta
        drone_cos_angle[j] = np.cos(theta)
        drone_sin_angle[j] = np.sin(theta)

    t_drop, fusetime = {}, {}
    for (j, k), (td, ft) in bombs_info.items():
        t_drop[(j, k)] = td
        fusetime[(j, k)] = ft
    
    # --- 2. 检查约束并计算惩罚 ---
    penalty = 0
    for j in drones_info.keys():
        drop_times_for_drone = sorted([t_drop[(j,k)] for k in range(num_bombs_per_drone)])
        for i in range(len(drop_times_for_drone) - 1):
            if drop_times_for_drone[i+1] - drop_times_for_drone[i] < MIN_DROP_INTERVAL:
                penalty += 20 # 给予重罚

    t_det = {(j,k): t_drop[(j,k)] + fusetime[(j,k)] for j, k in bombs}
    
    # --- 3. 模拟主循环 ---
    

    if verbose:
        for j in drones_info.keys():
            print(f"Drone {j} speed: {drone_speed[j]}, angle: {np.rad2deg(drone_angle[j]):.2f} degrees")
            for k in range(num_bombs_per_drone):
                print(f"Bomb {j},{k} drop time: {t_drop[(j,k)]}, fuse time: {fusetime[(j,k)]}")
                print(f"Bomb {j},{k} detonation time: {t_det[(j,k)]}")

    time_steps = np.arange(0, time_horizon, time_step)
    total_obscured_time = 0.0
    num_time_steps = len(time_steps)
    is_obscured = dict.fromkeys(range(num_time_steps), 0)
    
    # 主循环
    for t_idx, t in enumerate(time_steps):
        # This variable is 1 if all lines of sight are blocked at time t

        # Intermediate variables for positions
        # Missile positions at time t
        missile_pos = {}
        for i, p0_i in missiles_info.items():
            p0_vec = np.array(p0_i)
            dist_to_target = np.linalg.norm(p0_vec)
            time_to_target = dist_to_target / V_MISSILE
            if t < time_to_target:
                 missile_pos[i] = p0_vec * (1 - (V_MISSILE * t) / dist_to_target)
            else:
                 missile_pos[i] = np.array([0,0,0]) # Reached target
        
        p_drop_x = dict.fromkeys(bombs, 0)
        p_drop_y = dict.fromkeys(bombs, 0)
        p_det_x = dict.fromkeys(bombs, 0)
        p_det_y = dict.fromkeys(bombs, 0)
        p_det_z = dict.fromkeys(bombs, 0)
        p_cloud_x = dict.fromkeys(bombs, 0)
        p_cloud_y = dict.fromkeys(bombs, 0)
        p_cloud_z = dict.fromkeys(bombs, 0)

        for j, k in bombs:
            if t_det[j,k] > t or t > t_det[j,k] + SMOKE_DURATION:
                continue
            p_drop_x[j,k] = drones_info[j][0] + drone_speed[j] * drone_cos_angle[j] * t_drop[j,k]
            p_drop_y[j,k] = drones_info[j][1] + drone_speed[j] * drone_sin_angle[j] * t_drop[j,k]
            p_det_x[j,k] = p_drop_x[j,k] + drone_speed[j] * drone_cos_angle[j] * fusetime[j,k]
            p_det_y[j,k] = p_drop_y[j,k] + drone_speed[j] * drone_sin_angle[j] * fusetime[j,k]
            p_det_z[j,k] = drones_info[j][2] - 0.5 * G * fusetime[j,k] * fusetime[j,k]
            p_cloud_x[j,k] = p_det_x[j,k]
            p_cloud_y[j,k] = p_det_y[j,k]
            p_cloud_z[j,k] = p_det_z[j,k] - V_SINK * (t - t_det[j,k])


        for i in missiles_info.keys():
            for p_target in TARGET_POINTS:
                line_blocked = 0
                missile_A = missile_pos[i]
                target_B = p_target
                AB = target_B - missile_A 
                for j, k in bombs:
                    if t_det[j,k] > t or t > t_det[j,k] + SMOKE_DURATION:
                        continue

                    # 1. Define the line segment from missile (A) to target point (B)
                    cloud_C = np.array([p_cloud_x[j,k], p_cloud_y[j,k], p_cloud_z[j,k]])
                    AC = cloud_C - missile_A
                    BC = cloud_C - target_B
                    
                    # Project AC onto AB to find the closest point on the line AB
                    s = np.dot(AC, AB) / np.dot(AB, AB)
                    
                    dist_sq = 0
                    if s < 0:
                        # Closest point is A (missile)
                        dist_sq = np.dot(AC, AC)
                    elif s > 1:
                        # Closest point is B (target)
                        dist_sq = np.dot(BC, BC)
                    else:
                        # Closest point is on the segment, use projection
                        projection = missile_A + s * AB
                        dist_sq = np.dot(cloud_C - projection, cloud_C - projection)

                    # print(f" time: {t}, target_point: {p_target}, cloud_C {j,k}: {cloud_C}")

                    if dist_sq <= R_SMOKE * R_SMOKE:
                        line_blocked = 1
                        # if t < 6 :
                        #     print(f" time: {t}, target_point: {p_target}, cloud_C {j,k}: {cloud_C}")
                if line_blocked == 0:
                    break
            if line_blocked == 0:
                break
        is_obscured[t_idx] = line_blocked

        # print(f"Time {t}s 目标是否被遮挡: {is_obscured[t_idx]}")

    total_obscured_time = sum(is_obscured[t_idx] * time_step for t_idx in range(num_time_steps)) 

    return total_obscured_time - penalty

# --- 2. REINFORCE 策略网络 ---
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim) # 输出动作的均值
        )
        # 将标准差设置为可学习的参数，为每个动作维度设置独立的标准差
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        action_mean = self.net(obs)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        return dist

def pretrain_policy_with_expert(policy, optimizer, expert_action, observation, epochs=200):
    """
    使用专家动作对策略网络进行监督学习预训练。
    """
    print("\n--- 开始策略网络预训练 ---")
    
    expert_action_tensor = torch.tensor(expert_action, dtype=torch.float32)
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    for epoch in range(epochs):
        # 从策略网络获取当前动作的均值
        action_mean = policy(observation).mean
        
        # 计算损失
        loss = loss_fn(action_mean, expert_action_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"预训练 Epoch [{epoch+1}/{epochs}], 损失: {loss.item():.6f}")

    end_time = time.time()
    print(f"预训练完成！耗时: {end_time - start_time:.2f} 秒")

# --- 3. 主训练逻辑 ---
if __name__ == '__main__':
    # --- 问题定义 ---
    problem_missiles = { 'M1': (20000, 0, 2000) }
    problem_drones = { 'FY1': (17800, 0, 1800) }
    problem_num_bombs = 3
    problem_time_horizon = 20
    problem_time_step_train = 0.2
    problem_time_step_eval = 0.01

    # --- 环境和模型参数 ---
    num_drones = len(problem_drones)
    num_bombs = len(problem_drones) * problem_num_bombs
    action_dim = (num_drones * 2) + (num_bombs * 2)
    obs_dim = (len(problem_missiles) * 3) + (num_drones * 3)

    # 创建固定的观测值
    obs_list = []
    for pos in problem_missiles.values():
        obs_list.extend(pos)
    for pos in problem_drones.values():
        obs_list.extend(pos)
    observation = torch.tensor(obs_list, dtype=torch.float32)

    # --- 初始化策略网络和优化器 ---
    policy = Policy(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # --- NEW STEP: 定义专家策略并进行预训练 ---
    # 这是从 simulator.py 中提取的优秀解
    expert_flight_info = {
        'FY1': (139.99, np.pi * 179.65 / 180),
    }
    expert_bombs_info = {
        ('FY1', 0): (0, 3.61),
        ('FY1', 1): (3.66, 5.33),
        ('FY1', 2): (5.55, 6.06)
    }

    # 将专家策略编码为归一化的动作向量
    expert_action_normalized = encode_solution_to_action(
        expert_flight_info,
        expert_bombs_info,
        problem_drones,
        problem_num_bombs,
        problem_time_horizon
    )
    
    # 执行预训练
    pretrain_policy_with_expert(policy, optimizer, expert_action_normalized, observation, epochs=300)

    # --- 训练超参数 ---
    num_episodes = 20000
    
    print("\n--- 开始 REINFORCE 微调训练 ---")
    start_time = time.time()
    
    best_reward = -np.inf
    best_action = None

    for episode in range(num_episodes):
        # 从策略中采样一个动作
        dist = policy(observation)
        action = dist.sample() # 采样动作
        log_prob = dist.log_prob(action).sum() # 计算该动作的对数概率

        # 在环境中执行动作并获得奖励
        action_np = action.detach().numpy()
        flight_info, bombs_info = decode_action(action_np, problem_drones, problem_num_bombs, problem_time_horizon)
        
        reward = run_simulation(
            flight_info, bombs_info, problem_missiles, problem_drones,
            problem_num_bombs, problem_time_horizon, problem_time_step_train
        )
        
        # 检查预训练策略的初始奖励
        if episode == 0:
            print("--- 检查预训练策略的初始性能 ---")
            initial_action = policy(observation).mean.detach().numpy()
            init_flight_info, init_bombs_info = decode_action(initial_action, problem_drones, problem_num_bombs, problem_time_horizon)
            initial_reward = run_simulation(
                init_flight_info, init_bombs_info, problem_missiles, problem_drones,
                problem_num_bombs, problem_time_horizon, problem_time_step_train, verbose=False
            )
            print(f"预训练后策略的初始奖励: {initial_reward:.4f}")
            best_reward = initial_reward
            best_action = initial_action


        # 记录最优结果
        if reward > best_reward:
            best_reward = reward
            # 使用策略的均值作为确定性最优动作
            best_action = policy(observation).mean.detach().numpy()

        # 计算损失函数
        # REINFORCE 目标: max(log_prob * reward) -> min(-log_prob * reward)
        loss = -log_prob * reward
        
        # 更新策略网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode [{episode+1}/{num_episodes}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Reward: {reward:.4f}, "
                  f"Best Reward (Train): {best_reward:.4f}")

    end_time = time.time()
    print(f"\n训练完成！耗时: {end_time - start_time:.2f} 秒")

    # --- 使用找到的最优策略进行最终评估 ---
    print("\n\n--- 使用训练找到的最优策略进行最终评估 ---")
    if best_action is not None:
        final_flight_info, final_bombs_info = decode_action(
            best_action, problem_drones, problem_num_bombs, problem_time_horizon
        )

        print("\n--- 最优策略详情 ---")
        for j, (speed, angle) in final_flight_info.items():
            print(f"无人机 {j}:")
            print(f"  飞行速度: {speed:.2f} m/s")
            print(f"  飞行角度: {np.rad2deg(angle):.2f} 度")
        
        sorted_bombs = sorted(final_bombs_info.items(), key=lambda item: item[1][0])
        for (j, k), (t_drop, fusetime) in sorted_bombs:
            print(f"烟雾弹 ({j}, {k}):")
            print(f"  投掷时间: {t_drop:.2f} s")
            print(f"  引信时间: {fusetime:.2f} s")
            print(f"  -> 引爆时间: {t_drop + fusetime:.2f} s")

        print("\n--- 使用最优策略进行最终高精度模拟评估 ---")
        final_score = run_simulation(
            final_flight_info, final_bombs_info, problem_missiles, problem_drones,
            problem_num_bombs, problem_time_horizon, problem_time_step_eval, verbose=True
        )
        print("\n=============================================")
        print(f"最终评估 - 总有效遮蔽时间: {final_score:.2f} 秒")
        print("=============================================")
    else:
        print("未能找到有效策略。")
