# main.py
import gymnasium as gym
from simulation_env import SmokeStrategyEnv
from ppo_agent import PPO
import torch
import numpy as np
import os

def main():
    # --- 创建环境 ---
    # 确保simulation.py文件在同一个目录下
    env = SmokeStrategyEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- PPO 超参数 ---
    total_episodes = 5000         # 总训练回合数
    max_ep_len = 1                # 每个回合只有一个时间步
    update_timestep = 10          # 每10个回合更新一次网络
    
    lr_actor = 0.0003             # actor学习率
    lr_critic = 0.001             # critic学习率
    gamma = 0.99                  # 折扣因子 (在这个问题中作用不大)
    K_epochs = 40                 # 更新策略的epoch数
    eps_clip = 0.2                # PPO裁剪范围
    
    action_std = 0.6              # 初始动作标准差
    action_std_decay_rate = 0.05  # 标准差衰减率
    min_action_std = 0.1          # 最小标准差

    # --- 初始化PPO智能体 ---
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # --- 训练循环 ---
    print("--- Starting PPO Training ---")
    
    timestep = 0
    best_reward = -1
    best_action = None

    for i_episode in range(1, total_episodes + 1):
        state, _ = env.reset()
        
        # 在训练过程中逐渐减小探索噪声
        if i_episode % 100 == 0:
            new_std = ppo_agent.policy_old.action_var[0].item()**0.5 - action_std_decay_rate
            ppo_agent.set_action_std(max(new_std, min_action_std))

        # 收集数据
        action = ppo_agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        
        # 更新智能体
        if i_episode % update_timestep == 0:
            ppo_agent.update(reward)

        # 记录最优结果
        if reward > best_reward:
            best_reward = reward
            best_action = action

        if i_episode % 50 == 0:
            print(f"Episode {i_episode}, Last Reward: {reward:.2f}, Best Reward: {best_reward:.2f}")

    print("\n--- Training Finished ---")
    print(f"Best total obscuration time found: {best_reward:.2f}s")
    
    # 使用找到的最优策略（动作）来解码并打印
    print("\n--- Best Strategy Found ---")
    final_real_action = env._unscale_action(best_action)
    # ... 此处可以添加解码并打印具体策略参数的代码 ...
    
if __name__ == '__main__':
    # 确保simulation.py存在
    if not os.path.exists('simulation.py'):
        print("Error: simulation.py not found. Please ensure it is in the same directory.")
    else:
        main()