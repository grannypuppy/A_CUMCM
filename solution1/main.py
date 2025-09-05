# main.py
import numpy as np
from optimizer import find_best_continuous_strategy, solve_assignment_problem
from config import *
import time

def run_q5_solution():
    """
    完整解决问题5的流程
    """
    drone_ids = list(INITIAL_POSITIONS_DRONES.keys())
    missile_ids = list(INITIAL_POSITIONS_MISSILES.keys())
    
    num_drones = len(drone_ids)
    num_missiles = len(missile_ids)
    
    # --- 阶段一: 预计算价值矩阵 V[i, j, k] ---
    # V[i, j, k] = 无人机i用k+1枚弹攻击导弹j的最大遮蔽时间
    print("--- STAGE 1: Pre-computation of strategy values ---")
    print("This will take a significant amount of time...")
    
    # k=0: 1枚弹, k=1: 2枚弹, k=2: 3枚弹
    value_matrix = np.zeros((num_drones, num_missiles, MAX_DECOYS_PER_DRONE))
    
    start_time = time.time()
    
    for i, drone_id in enumerate(drone_ids):
        for j, missile_id in enumerate(missile_ids):
            for k in range(MAX_DECOYS_PER_DRONE):
                num_decoys = k + 1
                print(f"Optimizing for {drone_id} vs {missile_id} with {num_decoys} decoy(s)...")
                
                max_time, _ = find_best_continuous_strategy(drone_id, num_decoys, missile_id)
                value_matrix[i, j, k] = max_time
                
                print(f"  -> Achieved max obscuration time: {max_time:.2f}s")

    end_time = time.time()
    print(f"\nPre-computation finished in {end_time - start_time:.2f} seconds.")
    
    # --- 阶段二: Gurobi求解最优分配 ---
    print("\n--- STAGE 2: Solving assignment problem with Gurobi ---")
    
    assignment = solve_assignment_problem(value_matrix)
    
    # --- 阶段三: 输出最终结果 ---
    print("\n--- FINAL RESULT: Optimal Assignment ---")
    if not assignment:
        print("No assignment was found.")
    else:
        total_value = 0
        for drone_name, (missile_name, num_decoys) in assignment.items():
            print(f"Assign {drone_name} to attack {missile_name} using {num_decoys} decoy(s).")
            # 查找对应的价值
            i = drone_ids.index(drone_name)
            j = missile_ids.index(missile_name)
            k = num_decoys - 1
            value = value_matrix[i, j, k]
            total_value += value
            print(f"  - Expected obscuration time from this task: {value:.2f}s")
        print(f"\nTotal expected obscuration time (summed): {total_value:.2f}s")

    print("\nNote: To generate the final result3.xlsx, you would re-run the continuous optimizer")
    print("for each task in the final assignment to get the detailed flight and drop parameters,")
    print("and then format them into the required excel file.")


if __name__ == "__main__":
    run_q5_solution()