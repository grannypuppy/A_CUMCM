# main.py
import numpy as np
import time
from ga_optimizer import GeneticOptimizer
from config import INITIAL_POSITIONS_DRONES

def main():
    # GA 参数 (注意：高保真模型计算量极大，建议先用小种群和少代数测试)
    POPULATION_SIZE = 50      # 种群大小
    GENERATIONS = 100         # 迭代代数 (建议从50开始测试)
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1       # 稍高的变异率以增加探索

    print("--- Starting Holistic GA Solver (High-Fidelity Model) ---")
    print(f"Parameters: Population={POPULATION_SIZE}, Generations={GENERATIONS}\n")
    
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
    
    print("\n--- Best Strategy Found (Chromosome Genes) ---")
    drone_ids = list(INITIAL_POSITIONS_DRONES.keys())
    for i in range(len(drone_ids)):
        start_idx = i * 8
        genes = best_chromosome[start_idx : start_idx + 8]
        print(f"\nDrone {drone_ids[i]}:")
        print(f"  - Flight: Speed={genes[0]:.2f} m/s, Angle={np.rad2deg(genes[1]):.2f} degrees")
        print(f"  - Decoy 1: Drop at t={genes[2]:.2f}s, Detonate after {genes[3]:.2f}s")
        print(f"  - Decoy 2: Drop at t={genes[4]:.2f}s, Detonate after {genes[5]:.2f}s")
        print(f"  - Decoy 3: Drop at t={genes[6]:.2f}s, Detonate after {genes[7]:.2f}s")


if __name__ == "__main__":
    main()