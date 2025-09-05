# simulation_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import *
# 我们需要从您之前确认的GA方案中导入那个核心计算函数
from simulation import calculate_total_obscuration_time 

class SmokeStrategyEnv(gym.Env):
    def __init__(self):
        super(SmokeStrategyEnv, self).__init__()

        self.drone_ids = list(INITIAL_POSITIONS_DRONES.keys())
        self.missile_ids = list(INITIAL_POSITIONS_MISSILES.keys())
        self.num_drones = len(self.drone_ids)
        self.genes_per_drone = 8
        self.action_dim = self.num_drones * self.genes_per_drone # 40

        # 定义动作空间: 40个连续值，归一化到[-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

        # 定义状态空间: 所有对象的初始位置坐标拼接
        initial_states = []
        for missile_id in self.missile_ids:
            initial_states.extend(INITIAL_POSITIONS_MISSILES[missile_id])
        for drone_id in self.drone_ids:
            initial_states.extend(INITIAL_POSITIONS_DRONES[drone_id])
        self.state_dim = len(initial_states)
        self.initial_state = np.array(initial_states, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # 动作参数的真实边界
        self.real_bounds = [
            DRONE_SPEED_RANGE,          # speed
            (0, 2 * np.pi),             # angle
            (0.1, 80.0), (0.1, 20.0),  # t_drop1, t_fuze1
            (0.1, 80.0), (0.1, 20.0),  # t_drop2, t_fuze2
            (0.1, 80.0), (0.1, 20.0),  # t_drop3, t_fuze3
        ] * self.num_drones

    def _unscale_action(self, action):
        """将[-1, 1]的动作反归一化到真实物理范围"""
        unscaled_action = np.zeros_like(action)
        for i in range(self.action_dim):
            low, high = self.real_bounds[i]
            unscaled_action[i] = low + (action[i] + 1.0) * 0.5 * (high - low)
        return unscaled_action

    def step(self, action):
        """执行一个动作并返回结果"""
        # 1. 反归一化动作
        real_action = self._unscale_action(action)
        
        # 2. 解码动作为策略
        strategies = {missile_id: [] for missile_id in self.missile_ids}
        is_action_valid = True
        
        all_drone_strategies = []
        for i in range(self.num_drones):
            drone_idx_start = i * self.genes_per_drone
            drone_action = real_action[drone_idx_start : drone_idx_start + self.genes_per_drone]
            
            speed, angle = drone_action[0], drone_action[1]
            drone_id = self.drone_ids[i]

            # 检查投放时间约束
            drop_times = sorted([drone_action[2 + 2*j] for j in range(3)])
            if not all(drop_times[j+1] - drop_times[j] >= MIN_INTERVAL_DECOYS for j in range(len(drop_times)-1)):
                is_action_valid = False
                break
            
            for j in range(3): # 3枚弹
                t_drop = drone_action[2 + 2*j]
                t_fuze = drone_action[3 + 2*j]
                # 在这里，我们不区分导弹，将所有策略放在一起
                all_drone_strategies.append((drone_id, speed, angle, t_drop, t_fuze))
        
        # 3. 计算奖励（总遮蔽时间）
        if not is_action_valid:
            reward = 0.0 # 惩罚无效动作
        else:
            # 这是一个简化的计算，实际需要调用包含48个顶点检查的函数
            # 我们假设这个函数是存在的
            # total_time = calculate_obscuration_with_48_vertices(all_drone_strategies, self.missile_ids)
            
            # 为了代码能运行，我们先用一个模拟的总和
            total_reward = 0
            for missile_id in self.missile_ids:
                 # 注意：这里传入的是所有15枚弹的策略
                total_reward += calculate_total_obscuration_time(all_drone_strategies, missile_id)
            reward = total_reward

        # 4. 回合结束
        done = True
        # observation, reward, terminated, truncated, info
        return self.initial_state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        return self.initial_state, {}