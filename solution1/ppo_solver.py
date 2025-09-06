"""
使用近端策略优化 (PPO) 算法解决烟幕屏部署问题的强化学习求解器。

该脚本需要安装以下库:
pip install stable-baselines3[extra] gymnasium torch
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time
import torch

# --- 1. 从 simulator.py 中借鉴的核心模拟逻辑和常量 ---
# 将模拟器逻辑封装到 Gym 环境中

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
    target_points = []
    for h in [0, TARGET_HEIGHT]:
        for i in range(24):
            angle = 2 * np.pi * i / 24
            x = TARGET_CENTER_BASE[0] + TARGET_RADIUS * np.cos(angle)
            y = TARGET_CENTER_BASE[1] + TARGET_RADIUS * np.sin(angle)
            z = TARGET_CENTER_BASE[2] + h
            target_points.append(np.array([x, y, z]))
    return target_points

class SmokeScreenEnv(gym.Env):
    """
    一个为烟幕屏部署问题自定义的 Gym 环境。

    - 状态 (Observation): 固定的初始条件 (导弹和无人机的初始位置)。
    - 动作 (Action): 一个包含所有决策变量的归一化向量。
    - 奖励 (Reward): 总遮蔽时间减去对无效动作的惩罚。
    - 回合 (Episode): 每一回合只包含一步。智能体做出完整决策，环境运行完整模拟并返回最终奖励。
    """
    def __init__(self, missiles_info, drones_info, num_bombs_per_drone, time_horizon, time_step):
        super(SmokeScreenEnv, self).__init__()

        self.missiles_info = missiles_info
        self.drones_info = drones_info
        self.num_bombs_per_drone = num_bombs_per_drone
        self.time_horizon = time_horizon
        self.time_step = time_step
        
        self.bombs = [(j, k) for j in self.drones_info.keys() for k in range(self.num_bombs_per_drone)]
        self.target_points = generate_target_points()

        # --- 定义动作空间 (Action Space) ---
        # 动作向量的每个元素都在 [-1, 1] 之间，需要后续映射到真实物理值
        # 动作组成: [drone_speed, drone_angle, bomb1_tdrop, bomb1_ftime, bomb2_tdrop, bomb2_ftime, ...]
        num_drones = len(drones_info)
        action_dim = (num_drones * 2) + (len(self.bombs) * 2) 
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

        # --- 定义状态空间 (Observation Space) ---
        # 状态是固定的初始条件
        obs_dim = (len(self.missiles_info) * 3) + (len(self.drones_info) * 3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.initial_obs = self._get_initial_obs()

    def _get_initial_obs(self):
        obs = []
        for pos in self.missiles_info.values():
            obs.extend(pos)
        for pos in self.drones_info.values():
            obs.extend(pos)
        return np.array(obs, dtype=np.float32)

    def _decode_action(self, action):
        """将归一化的动作向量 [-1, 1] 映射到真实的物理值范围"""
        decoded = {}
        flight_info = {}
        bombs_info = {}
        
        action_idx = 0
        
        # 1. 解码无人机飞行参数
        for drone_id in self.drones_info.keys():
            # 速度: [70, 140] m/s
            speed = 105 + action[action_idx] * 35
            action_idx += 1
            # 角度: [0, 2*pi] rad
            angle = np.pi + action[action_idx] * np.pi
            action_idx += 1
            flight_info[drone_id] = (speed, angle)

        # 2. 解码每个烟雾弹的部署参数
        for bomb_id in self.bombs:
            # 投掷时间: [0, time_horizon/2] 
            # 限制投掷时间在前半段以确保有足够时间发挥作用
            t_drop = (self.time_horizon / 4) + action[action_idx] * (self.time_horizon / 4)
            action_idx += 1
            # 引信时间: [1, 10] s (一个合理的范围)
            fusetime = 5.5 + action[action_idx] * 4.5
            action_idx += 1
            bombs_info[bomb_id] = (t_drop, fusetime)
            
        decoded['flight_info'] = flight_info
        decoded['bombs_info'] = bombs_info
        return decoded

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.initial_obs, {}

    def step(self, action):
        decoded_action = self._decode_action(action)
        flight_info = decoded_action['flight_info']
        bombs_info = decoded_action['bombs_info']

        # --- 运行模拟计算总遮蔽时间 (核心奖励) ---
        total_coverage, penalty = self.run_simulation(flight_info, bombs_info)
        
        # --- 计算最终奖励 ---
        reward = total_coverage - penalty

        # 回合结束
        terminated = True
        truncated = False
        
        return self.initial_obs, reward, terminated, truncated, {}

    def run_simulation(self, flight_info, bombs_info):
        """根据给定的决策执行一次完整的物理模拟"""
        
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
        for j in self.drones_info.keys():
            drop_times_for_drone = sorted([t_drop[(j,k)] for k in range(self.num_bombs_per_drone)])
            for i in range(len(drop_times_for_drone) - 1):
                if drop_times_for_drone[i+1] - drop_times_for_drone[i] < MIN_DROP_INTERVAL:
                    penalty += 50 # 对违反最小投弹间隔的行为进行惩罚

        t_det = {(j,k): t_drop[(j,k)] + fusetime[(j,k)] for j, k in self.bombs}
        
        # --- 3. 模拟主循环 ---
        time_steps = np.arange(0, self.time_horizon, self.time_step)
        num_time_steps = len(time_steps)
        total_obscured_time = 0.0

        for t_idx, t in enumerate(time_steps):
            missile_pos = {}
            for i, p0_i in self.missiles_info.items():
                p0_vec = np.array(p0_i)
                target_vec = TARGET_CENTER_BASE
                direction_vec = target_vec - p0_vec
                dist_to_target = np.linalg.norm(direction_vec)
                if dist_to_target > 0:
                    time_to_target = dist_to_target / V_MISSILE
                    missile_pos[i] = p0_vec + direction_vec * (V_MISSILE * t / dist_to_target) if t < time_to_target else target_vec
                else:
                    missile_pos[i] = target_vec

            is_target_obscured_this_step = False
            for i in self.missiles_info.keys():
                for p_target in self.target_points:
                    line_blocked = False
                    missile_A = missile_pos[i]
                    target_B = p_target
                    AB_sq = np.dot(target_B - missile_A, target_B - missile_A)
                    if AB_sq == 0: continue

                    for j, k in self.bombs:
                        if not (t_det[(j,k)] <= t <= t_det[(j,k)] + SMOKE_DURATION):
                            continue

                        p_drop_x = self.drones_info[j][0] + drone_speed[j] * drone_cos_angle[j] * t_drop[(j,k)]
                        p_drop_y = self.drones_info[j][1] + drone_sin_angle[j] * t_drop[(j,k)]
                        p_det_z = self.drones_info[j][2] - 0.5 * G * fusetime[(j,k)]**2
                        p_det_x = p_drop_x + drone_speed[j] * drone_cos_angle[j] * fusetime[(j,k)]
                        p_det_y = p_drop_y + drone_sin_angle[j] * drone_sin_angle[j] * fusetime[(j,k)]
                        
                        p_cloud_center_z = p_det_z - V_SINK * (t - t_det[(j,k)])
                        cloud_C = np.array([p_det_x, p_det_y, p_cloud_center_z])

                        AC = cloud_C - missile_A
                        s = np.dot(AC, target_B - missile_A) / AB_sq
                        
                        dist_sq = 0
                        if s < 0: dist_sq = np.dot(AC, AC)
                        elif s > 1: dist_sq = np.dot(cloud_C - target_B, cloud_C - target_B)
                        else: dist_sq = np.dot(AC, AC) - s * s * AB_sq
                        
                        if dist_sq <= R_SMOKE**2:
                            line_blocked = True
                            break
                    
                    if line_blocked:
                        is_target_obscured_this_step = True
                        break
                if is_target_obscured_this_step:
                    break
            
            if is_target_obscured_this_step:
                total_obscured_time += self.time_step
        
        return total_obscured_time, penalty

# --- NEW FUNCTION: To encode a known solution into a normalized action vector ---
def encode_solution_to_action(flight_info, bombs_info, drones_info, num_bombs_per_drone, time_horizon):
    """
    将一个已知的决策方案 (物理单位) 编码为归一化的动作向量 [-1, 1].
    这是 SmokeScreenEnv._decode_action 方法的逆操作。
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

# --- NEW FUNCTION: To pre-train the agent's policy network ---
def pretrain_agent_with_expert(model, expert_action, initial_obs, epochs=100, learning_rate=3e-4):
    """
    使用专家动作对 PPO 模型的策略网络进行监督学习预训练。
    """
    print("\n--- 开始策略网络预训练 ---")
    
    # 转换为 PyTorch 张量
    expert_action_tensor = torch.tensor(expert_action, dtype=torch.float32).unsqueeze(0)
    initial_obs_tensor = torch.tensor(initial_obs, dtype=torch.float32).unsqueeze(0)
    
    # 获取策略网络和优化器
    policy = model.policy
    optimizer = policy.optimizer
    
    # 定义损失函数
    loss_fn = torch.nn.MSELoss()
    
    start_time = time.time()
    for epoch in range(epochs):
        # 从策略网络获取当前动作的均值
        # 注意: 我们只关心均值，因为这是策略的核心输出
        obs_features = policy.extract_features(initial_obs_tensor)
        action_mean = policy.action_net(policy.mlp_extractor.policy_net(obs_features))

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


if __name__ == '__main__':
    # --- 问题定义 ---
    problem_missiles = { 'M1': (20000, 0, 2000) }
    problem_drones = { 'FY1': (17800, 0, 1800) }
    problem_num_bombs = 3
    problem_time_horizon = 20
    problem_time_step = 0.2  # 训练时使用较粗的时间步以加速

    # --- 1. 创建环境 ---
    env = SmokeScreenEnv(
        missiles_info=problem_missiles,
        drones_info=problem_drones,
        num_bombs_per_drone=problem_num_bombs,
        time_horizon=problem_time_horizon,
        time_step=problem_time_step
    )
    # 检查环境是否符合 Gym API
    print("--- 检查环境 ---")
    check_env(env)
    print("环境检查通过！")

    # --- 2. 创建 PPO 模型 (此时策略是随机初始化的) ---
    log_dir = "./ppo_smokescreen_logs/"
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        gamma=0.995,       # 在单步环境中 gamma 影响不大，但习惯上设置
        n_steps=2048,      # 每次更新前收集的数据量
        batch_size=64,
        n_epochs=10,
        gae_lambda=0.95,
        ent_coef=0.01,     # 鼓励探索的熵系数
    )

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
    pretrain_agent_with_expert(model, expert_action_normalized, env.initial_obs, epochs=200)


    # --- 3. 开始标准的强化学习训练 (从预训练好的策略开始) ---
    print("\n--- 开始强化学习微调训练 ---")
    start_time = time.time()
    # 增加训练步数以获得更好的结果，例如 100000 或更高
    model.learn(total_timesteps=50000)
    end_time = time.time()
    print(f"训练完成！耗时: {end_time - start_time:.2f} 秒")
    model.save("ppo_smokescreen_model")

    # --- 4. 使用训练好的模型进行预测并展示结果 ---
    print("\n\n--- 使用训练好的模型寻找最优策略 ---")
    obs, _ = env.reset()
    action, _states = model.predict(obs, deterministic=True)

    decoded_solution = env._decode_action(action)
    final_flight_info = decoded_solution['flight_info']
    final_bombs_info = decoded_solution['bombs_info']

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
        
    # --- 5. 使用最优策略运行一次高精度模拟 ---
    print("\n--- 使用最优策略进行最终高精度模拟评估 ---")
    eval_env = SmokeScreenEnv(
        missiles_info=problem_missiles,
        drones_info=problem_drones,
        num_bombs_per_drone=problem_num_bombs,
        time_horizon=problem_time_horizon,
        time_step=0.01  # 使用更高的时间精度
    )
    final_score, _ = eval_env.run_simulation(final_flight_info, final_bombs_info)
    print("\n=============================================")
    print(f"最终评估 - 总有效遮蔽时间: {final_score:.2f} 秒")
    print("=============================================") 