# ga_optimizer.py
# (内容与上一回答中的版本一致，此处不再重复)
import numpy as np
import random
from config import *
from simulation import calculate_total_obscuration_time

class GeneticOptimizer:
    def __init__(self, population_size, generations, crossover_rate, mutation_rate, elitism_rate=0.05):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_elites = int(population_size * elitism_rate)

        self.drone_ids = list(INITIAL_POSITIONS_DRONES.keys())
        self.missile_ids = list(INITIAL_POSITIONS_MISSILES.keys())
        self.num_drones = len(self.drone_ids)
        self.num_missiles = len(self.missile_ids)
        
        # 定义染色体结构和边界
        # 每个无人机的基因: [target_missile_idx, num_decoys, speed, angle, t_drop1, t_fuze1, t_drop2, t_fuze2, t_drop3, t_fuze3]
        # target_missile_idx: -1 (不用), 0 (M1), 1 (M2), 2 (M3)
        self.genes_per_drone = 10
        self.chromosome_length = self.num_drones * self.genes_per_drone
        self.bounds = [
            (-1, self.num_missiles - 1), # target_missile_idx
            (1, MAX_DECOYS_PER_DRONE),   # num_decoys
            DRONE_SPEED_RANGE,          # speed
            (0, 2 * np.pi),             # angle
            (0.1, 80.0), (0.1, 20.0),  # t_drop1, t_fuze1
            (0.1, 80.0), (0.1, 20.0),  # t_drop2, t_fuze2
            (0.1, 80.0), (0.1, 20.0),  # t_drop3, t_fuze3
        ] * self.num_drones

    def _create_individual(self):
        """随机创建一个染色体"""
        individual = []
        for i in range(self.chromosome_length):
            bound = self.bounds[i]
            # 整数基因
            if i % self.genes_per_drone in [0, 1]:
                individual.append(random.randint(bound[0], bound[1]))
            # 浮点数基因
            else:
                individual.append(random.uniform(bound[0], bound[1]))
        return individual

    def _decode_chromosome(self, chromosome):
        """将染色体解码为策略"""
        strategies = {missile_id: [] for missile_id in self.missile_ids}
        for i in range(self.num_drones):
            drone_idx_start = i * self.genes_per_drone
            drone_genes = chromosome[drone_idx_start : drone_idx_start + self.genes_per_drone]
            
            target_missile_idx = int(drone_genes[0])
            if target_missile_idx == -1:
                continue

            num_decoys = int(drone_genes[1])
            speed, angle = drone_genes[2], drone_genes[3]
            drone_id = self.drone_ids[i]
            missile_id = self.missile_ids[target_missile_idx]
            
            # 检查投放时间约束
            drop_times = sorted([drone_genes[4 + 2*j] for j in range(num_decoys)])
            is_valid = all(drop_times[j+1] - drop_times[j] >= MIN_INTERVAL_DECOYS for j in range(len(drop_times)-1))
            if not is_valid: # 如果不满足，则此策略无效，适应度为0
                return None

            for j in range(num_decoys):
                t_drop = drone_genes[4 + 2*j]
                t_fuze = drone_genes[5 + 2*j]
                strategies[missile_id].append((drone_id, speed, angle, t_drop, t_fuze))
        
        return strategies

    def _calculate_fitness(self, chromosome):
        """计算染色体的适应度（总遮蔽时间）"""
        strategies_by_missile = self._decode_chromosome(chromosome)
        
        if strategies_by_missile is None:
            return 0 # 惩罚无效的投放间隔

        total_fitness = 0
        for missile_id, strategies in strategies_by_missile.items():
            total_fitness += calculate_total_obscuration_time(strategies, missile_id)
            
        return total_fitness

    def _selection(self, population, fitnesses):
        """锦标赛选择"""
        selected = []
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            winner = i if fitnesses[i] >= fitnesses[j] else j
            selected.append(population[winner])
        return selected

    def _crossover(self, parent1, parent2):
        """两点交叉"""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        
        pt1, pt2 = sorted(random.sample(range(self.chromosome_length), 2))
        child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
        return child1, child2

    def _mutate(self, individual):
        """对每个基因进行变异"""
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                bound = self.bounds[i]
                # 整数基因
                if i % self.genes_per_drone in [0, 1]:
                    individual[i] = random.randint(bound[0], bound[1])
                # 浮点数基因
                else:
                    # 高斯变异
                    mu = individual[i]
                    sigma = (bound[1] - bound[0]) * 0.1 # 变异步长
                    new_val = random.gauss(mu, sigma)
                    # 确保在边界内
                    individual[i] = max(bound[0], min(bound[1], new_val))
        return individual

    def run(self):
        """运行遗传算法"""
        # 1. 初始化种群
        population = [self._create_individual() for _ in range(self.population_size)]
        
        best_overall_fitness = -1
        best_overall_chromosome = None

        for gen in range(self.generations):
            # 2. 计算适应度
            fitnesses = [self._calculate_fitness(ind) for ind in population]
            
            # 记录当代和历史最优
            best_current_fitness = max(fitnesses)
            best_current_idx = np.argmax(fitnesses)
            if best_current_fitness > best_overall_fitness:
                best_overall_fitness = best_current_fitness
                best_overall_chromosome = population[best_current_idx][:]

            print(f"Generation {gen+1}/{self.generations} - Best Fitness: {best_overall_fitness:.2f}s")
            
            # 3. 生成新一代
            new_population = []
            
            # 精英主义：保留最好的个体
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(self.num_elites):
                new_population.append(population[sorted_indices[i]][:])

            # 4. 选择、交叉、变异
            selected_population = self._selection(population, fitnesses)
            
            for i in range(self.num_elites, self.population_size, 2):
                parent1, parent2 = random.sample(selected_population, 2)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(self._mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(child2))
            
            population = new_population

        return best_overall_chromosome, best_overall_fitness