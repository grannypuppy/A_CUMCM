# ga_optimizer.py
import numpy as np
import random
from tqdm import tqdm
from config import *
from simulation import calculate_fitness

class GeneticOptimizer:
    def __init__(self, population_size, generations, crossover_rate, mutation_rate, elitism_rate=0.05):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_elites = int(population_size * elitism_rate)

        self.num_drones = len(INITIAL_POSITIONS_DRONES)
        self.genes_per_drone = 8
        self.chromosome_length = self.num_drones * self.genes_per_drone
        
        # 定义染色体结构和边界
        # 每个无人机: [speed, angle, t_drop1, t_fuze1, t_drop2, t_fuze2, t_drop3, t_fuze3]
        drone_bounds = [
            DRONE_SPEED_RANGE,          # speed
            (0, 2 * np.pi),             # angle
            (0.1, 60.0), (0.1, 20.0),    # t_drop1, t_fuze1
            (0.1, 60.0), (0.1, 20.0),    # t_drop2, t_fuze2
            (0.1, 60.0), (0.1, 20.0),    # t_drop3, t_fuze3
        ]
        self.bounds = drone_bounds * self.num_drones

    def _create_individual(self):
        """随机创建一个染色体"""
        return [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.chromosome_length)]

    def _selection(self, population, fitnesses):
        """锦标赛选择"""
        selected = []
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            winner_idx = i if fitnesses[i] >= fitnesses[j] else j
            selected.append(population[winner_idx])
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
        """高斯变异"""
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                bound = self.bounds[i]
                sigma = (bound[1] - bound[0]) * 0.1
                new_val = random.gauss(individual[i], sigma)
                individual[i] = max(bound[0], min(bound[1], new_val))
        return individual

    def run(self):
        population = [self._create_individual() for _ in range(self.population_size)]
        best_overall_fitness = -1
        best_overall_chromosome = None

        for gen in tqdm(range(self.generations), desc="Evolving Generations"):
            fitnesses = [calculate_fitness(ind) for ind in population]
            
            best_current_fitness = max(fitnesses)
            if best_current_fitness > best_overall_fitness:
                best_overall_fitness = best_current_fitness
                best_overall_chromosome = population[np.argmax(fitnesses)][:]
            
            tqdm.write(f"Generation {gen+1} | Best Fitness: {best_overall_fitness:.2f}s | Current Gen Best: {best_current_fitness:.2f}s")
            
            new_population = []
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(self.num_elites):
                new_population.append(population[sorted_indices[i]][:])

            selected_population = self._selection(population, fitnesses)
            
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(selected_population, 2)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(c2))
            
            population = new_population

        return best_overall_chromosome, best_overall_fitness