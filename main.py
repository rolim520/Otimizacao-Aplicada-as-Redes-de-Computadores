"""
Calculo do tempo de execucao de um problema da mochila
Aluno: Guilherme Oliveira Rolim Silva
Implementação original com solver PuLP: Rodrigo de Souza Couto - GTA/PEE/COPPE/UFRJ

"""

import time as tm
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from scipy.stats import norm
import pickle
from pulp import *

#This mode saves the LP files (it only saves one, overwriting the others)
DEBUG_MODE = True

def knapsack_problem_solver(items, capacity, gap_threshold=0):
    # Creates the problem
    problem = LpProblem("Knapsack_Problem", LpMaximize)

    # Creates the decision variables
    x = LpVariable.dicts("Item", [i for i, _, _ in items], 0, 1, LpBinary)

    # Adds the objective function
    problem += lpSum([value * x[item] for item, _, value in items])

    # Adds the capacity restriction function
    problem += lpSum([weight * x[item] for item, weight, _ in items]) <= capacity

    if DEBUG_MODE:
        # Write the LP file
        problem.writeLP("knapsack.lp")

    # Solves the problem
    start_time = tm.time()
    problem.solve(PULP_CBC_CMD(gapRel=gap_threshold, msg=False))
    solve_time = tm.time() - start_time

    return value(problem.objective), solve_time

def algoritmo_guloso(items, capacity):

    start_time = tm.time()

    # --------------------------------------------------
    
    solution = []
    ordered_items = sorted(items, key=lambda i: i[2] / i[1], reverse=True)

    total_value = 0
    current_capacity = 0
    for i in range(len(ordered_items)):
        next_capacity = current_capacity + ordered_items[i][1]
        if next_capacity > capacity:
            break
        total_value += ordered_items[i][2]
        current_capacity = next_capacity
        solution.append(ordered_items[i])

    # --------------------------------------------------
    
    solve_time = tm.time() - start_time
    return total_value, solve_time


#For each run, this function generates random weights and values for the items
def generate_random_items(num_items, max_weight, max_value):
    return [(i, random.randint(1, max_weight), random.randint(1, max_value)) for i in range(num_items)]

def run_multiple_times(solver_func, items, capacity, num_runs, **kwargs):
    times = []
    objetivos = []
    
    for _ in range(num_runs):
        obj_val, solve_time = solver_func(items, capacity, **kwargs)
        times.append(solve_time)
        objetivos.append(obj_val)
        
    return np.mean(times), np.std(times), np.mean(objetivos)

#This function receives the results and plot a graph with time vs size
def plot_partial_results(results, sizes, num_runs, confidence=0.95):
    plt.figure(figsize=(10, 6))
    plt.xscale('log')
    for gap, data in results.items():
        if data:
            sizes, means, stds = zip(*data)
            import sys
            #Evaluates the score to use in a confidence interval
            #If there are more than 30 samples, we use normal distribution
            #Else, we use a t-student
            if num_runs >= 30:
                score = norm.ppf((1 + confidence) / 2)
            else:
                score = t_dist.ppf((1 + confidence) / 2, num_runs - 1)
            #In this case, they are calculated as 1.96 times the standard deviation divided by the square
            #root of the number of runs (num_runs). This represents a 95% confidence interval using a normal distribution.
            nome_legenda = 'Guloso' if gap == 'Guloso' else f'Gap {gap}'
            plt.errorbar(sizes, means, yerr=score * np.array(stds) / np.sqrt(num_runs), marker='o', label=nome_legenda)

    plt.title('Knapsack Problem: Execution time vs problem size for different gap values')
    plt.xlabel('Problem Size')
    plt.ylabel('Execution Time (s)')
    plt.grid(False)
    plt.legend()
    plt.xticks(sizes)
    plt.savefig('knapsack_execution_time.png')
    plt.close()

def plot_gap_objetivo(sizes, diferencas):
    plt.figure(figsize=(10, 6))
    plt.xscale('log')
    
    plt.plot(sizes, diferencas, marker='o', color='red', label='Erro do Guloso')
    
    plt.title('Knapsack Problem: Diferença da Função Objetivo (PuLP Ótimo vs Guloso)')
    plt.xlabel('Problem Size')
    plt.ylabel('Gap para o Ótimo (%)')
    plt.grid(True)
    plt.legend()
    plt.xticks(sizes, labels=[str(s) for s in sizes])
    
    # Salva o gráfico em PNG
    plt.savefig('knapsack_gap_objetivo.png')
    plt.close()

def main():
    #Problem sizes to test: number of items
    #sizes = [10, 1000, 10000, 100000, 1000000, 10000000]
    sizes = [10, 100, 1000, 10000, 100000]
    max_weight = 50
    max_value = 100
    gap_thresholds = [0.4, 0.2, 0.1, 0.05, 0.02, 0]
    num_runs = 10
    
    results = {gap: [] for gap in gap_thresholds}
    results['Guloso'] = []
    
    lista_de_diferencas = [] 

    for size in sizes:
        items = generate_random_items(size, max_weight, max_value)
        capacity = size * max_weight / 2 

        print(f"\nRodando Guloso para tamanho {size}...")
        mean_time_g, std_time_g, obj_guloso = run_multiple_times(algoritmo_guloso, items, capacity, num_runs)
        results['Guloso'].append((size, mean_time_g, std_time_g))

        obj_otimo_pulp = 0

        for gap in gap_thresholds:
            print(f"Rodando PuLP (Gap {gap}) para tamanho {size}...")
            mean_time, std_time, obj_pulp = run_multiple_times(knapsack_problem_solver, items, capacity, num_runs, gap_threshold=gap)
            results[gap].append((size, mean_time, std_time))
            
            if gap == 0:
                obj_otimo_pulp = obj_pulp
        
        diferenca = ((obj_otimo_pulp - obj_guloso) / obj_otimo_pulp) * 100
        print(f"-> Gap de Qualidade para tamanho {size}: {diferenca}")
        lista_de_diferencas.append(diferenca)

    plot_partial_results(results, sizes, num_runs)
    plot_gap_objetivo(sizes, lista_de_diferencas)

    with open('knapsack_results.pickle', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
