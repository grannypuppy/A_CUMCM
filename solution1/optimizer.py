# optimizer.py
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import differential_evolution
from simulation import calculate_total_obscuration_time
from config import *

# --- 底层连续优化器 ---
def find_best_continuous_strategy(drone_id, num_decoys, missile_id):
    """
    使用差分进化算法为单个任务找到最优的连续参数。
    任务定义：drone_id 使用 num_decoys 数量的弹药攻击 missile_id。
    返回 (最大遮蔽时间, 最优策略参数)
    """
    if num_decoys == 0:
        return 0.0, []

    # 决策变量x的边界: [v, theta, t_drop1, t_fuze1, t_drop2, t_fuze2, ...]
    bounds = [DRONE_SPEED_RANGE, (0, 2 * np.pi)] # v, theta
    for _ in range(num_decoys):
        bounds.extend([(0.1, 80.0), (0.1, 20.0)]) # t_drop, t_fuze

    # 目标函数 (需要最小化，所以返回负值)
    def objective_func(x):
        v, theta = x[0], x[1]
        
        # 检查投放间隔约束
        drop_times = sorted([x[2 + 2*i] for i in range(num_decoys)])
        for i in range(len(drop_times) - 1):
            if drop_times[i+1] - drop_times[i] < MIN_INTERVAL_DECOYS:
                return 1000 # 惩罚项，返回一个很大的正值（因为是最小化）

        strategies = []
        for i in range(num_decoys):
            t_drop = x[2 + 2*i]
            t_fuze = x[3 + 2*i]
            strategies.append((drone_id, v, theta, t_drop, t_fuze))

        return -calculate_total_obscuration_time(strategies, missile_id)

    # 使用差分进化算法求解
    result = differential_evolution(objective_func, bounds, strategy='best1bin', maxiter=100, popsize=20) # 适当调整参数

    # 解析结果
    max_time = -result.fun
    best_params = result.x
    
    # 将最优参数格式化为策略列表
    best_strategies = []
    v, theta = best_params[0], best_params[1]
    for i in range(num_decoys):
        t_drop = best_params[2 + 2*i]
        t_fuze = best_params[3 + 2*i]
        best_strategies.append((drone_id, v, theta, t_drop, t_fuze))

    return max_time, best_strategies

# --- 高层 Gurobi 优化器 ---
def solve_assignment_problem(value_matrix):
    """
    使用Gurobi求解任务分配问题。
    value_matrix[i, j, k] = 无人机i用k+1枚弹攻击导弹j能产生的最大遮蔽时间
    """
    num_drones = value_matrix.shape[0]
    num_missiles = value_matrix.shape[1]
    num_decoy_options = value_matrix.shape[2] # k from 0 to MAX_DECOYS_PER_DRONE-1

    model = gp.Model("Smoke_Strategy_Assignment")

    # 决策变量: x[i, j, k] = 1 如果无人机i用k+1枚弹攻击导弹j
    x = model.addVars(num_drones, num_missiles, num_decoy_options, vtype=GRB.BINARY, name="x")

    # 目标函数: 最大化总遮蔽时间
    # 注意：这里假设不同导弹的遮蔽时间是独立可加的，这是一个合理的简化
    model.setObjective(gp.quicksum(value_matrix[i, j, k] * x[i, j, k]
                                   for i in range(num_drones)
                                   for j in range(num_missiles)
                                   for k in range(num_decoy_options)),
                       GRB.MAXIMIZE)

    # 约束1: 每架无人机最多只能执行一个任务
    for i in range(num_drones):
        model.addConstr(gp.quicksum(x[i, j, k]
                                     for j in range(num_missiles)
                                     for k in range(num_decoy_options)) <= 1, name=f"drone_assign_{i}")

    # 求解模型
    model.optimize()

    # 解析并返回结果
    assignment = {}
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found!")
        solution = model.getAttr('x', x)
        for i in range(num_drones):
            for j in range(num_missiles):
                for k in range(num_decoy_options):
                    if solution[i, j, k] > 0.5:
                        drone_name = f"FY{i+1}"
                        missile_name = f"M{j+1}"
                        num_decoys = k + 1
                        assignment[drone_name] = (missile_name, num_decoys)
    else:
        print("No optimal solution found.")

    return assignment