# main.py
from ga_optimizer import GeneticOptimizer
import time
import numpy as np # For np.rad2deg

def main():
    """
    主函数：配置并运行基于48点精确模型的遗传算法。
    
    *** 重要提示 ***
    本模型采用了高保真度的48点遮蔽判断，计算量巨大。
    为在合理时间内得到结果，可考虑：
    1. 减少种群大小 (POPULATION_SIZE)
    2. 减少迭代代数 (GENERATIONS)
    3. 增大 config.py 中的 SIMULATION_TIME_STEP (e.g., to 0.2 or 0.25)
    """
    # GA 参数 (为演示目的设置得较小，实际应用中建议增大)
    POPULATION_SIZE = 50     # 种群大小
    GENERATIONS = 100        # 迭代代数
    CROSSOVER_RATE = 0.8     # 交叉概率
    MUTATION_RATE = 0.05     # 变异概率

    print("--- Starting GA with High-Fidelity 48-Point Obscuration Model ---")
    print(f"Parameters: Population={POPULATION_SIZE}, Generations={GENERATIONS}")
    print("WARNING: This process will be computationally intensive and may take a long time.")
    
    # 实例化并运行优化器
    optimizer = GeneticOptimizer(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE
    )
    
    start_time = time.time()
    best_chromosome, best_fitness = optimizer.run()
    end_time = time.time()
    
    print("\n--- Optimization Finished ---")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"Best overall fitness (Total Obscuration Time): {best_fitness:.2f}s")
    
    # 解析并打印最优策略
    print("\n--- Best Strategy Found (based on 48-Point Model) ---")
    final_strategies = optimizer._decode_chromosome(best_chromosome)
    
    if final_strategies is None:
        print("The best found chromosome was invalid (e.g., drop time constraint violation).")
        return

    for missile_id, strategies in final_strategies.items():
        if strategies:
            print(f"\nTasks for Missile {missile_id}:")
            for strat in strategies:
                drone_id, speed, angle, t_drop, t_fuze = strat
                print(f"  - Drone {drone_id}:")
                print(f"    - Flight: Speed={speed:.2f} m/s, Angle={np.rad2deg(angle):.2f} degrees")
                print(f"    - Deploy: Drop at t={t_drop:.2f}s, Detonate after {t_fuze:.2f}s")

if __name__ == "__main__":
    main()