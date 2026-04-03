"""
Calculo do tempo de execucao de um problema da mochila
Aluno: Guilherme Oliveira Rolim Silva
Implementação original com solver PuLP: Rodrigo de Souza Couto - GTA/PEE/COPPE/UFRJ
"""

import time as tm
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from scipy.stats import norm
import pickle
from pulp import *

DEBUG_MODE = False

def knapsack_problem_solver(items, capacity, gap_threshold=0):
    problem = LpProblem("Knapsack_Problem", LpMaximize)
    x = LpVariable.dicts("Item", [i for i, _, _ in items], 0, 1, LpBinary)
    problem += lpSum([value * x[item] for item, _, value in items])
    problem += lpSum([weight * x[item] for item, weight, _ in items]) <= capacity

    if DEBUG_MODE:
        problem.writeLP("knapsack.lp")

    start_time = tm.time()
    problem.solve(PULP_CBC_CMD(gapRel=gap_threshold, msg=False))
    solve_time = tm.time() - start_time

    return value(problem.objective), solve_time

def algoritmo_guloso(items, capacity):
    start_time = tm.time()
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

    solve_time = tm.time() - start_time
    return total_value, solve_time

def obter_vizinho(mochila, items, capacity, peso_atual, valor_atual):
    vizinho = mochila.copy()
    
    while True:
        indice = random.randint(0, len(vizinho) - 1)
        
        if vizinho[indice] == 1:
            vizinho[indice] = 0
            novo_peso = peso_atual - items[indice][1] 
            novo_valor = valor_atual - items[indice][2]
            return vizinho, novo_peso, novo_valor
            
        else:
            if peso_atual + items[indice][1] <= capacity:
                vizinho[indice] = 1
                novo_peso = peso_atual + items[indice][1]
                novo_valor = valor_atual + items[indice][2]
                return vizinho, novo_peso, novo_valor

def simulated_annealing(items, capacity, taxa_de_resfriamento=0.98, temp_inicial=1000, temp_final=1, max_iter=200):
    start_time = tm.time()
    mochila_atual = [0] * len(items)
    peso_atual = 0
    valor_atual = 0
    melhor_valor = 0

    temperatura = temp_inicial

    while temperatura > temp_final:
        for _ in range(max_iter):
            vizinho, peso_vizinho, valor_vizinho = obter_vizinho(mochila_atual, items, capacity, peso_atual, valor_atual)
            delta = valor_vizinho - valor_atual

            if delta > 0:
                mochila_atual = vizinho
                peso_atual = peso_vizinho
                valor_atual = valor_vizinho
                if valor_atual > melhor_valor:
                    melhor_valor = valor_atual
            else:
                if random.random() < math.exp(delta / temperatura):
                    mochila_atual = vizinho
                    peso_atual = peso_vizinho
                    valor_atual = valor_vizinho

        temperatura *= taxa_de_resfriamento

    solve_time = tm.time() - start_time
    return melhor_valor, solve_time

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

# Gráfico 1: Linhas exclusivas para os Gaps do PuLP
def plot_partial_results(results, sizes, num_runs, confidence=0.95):
    plt.figure(figsize=(10, 6))
    plt.xscale('log')
    for gap, data in results.items():
        if gap in ['Guloso', 'SA']: 
            continue
        
        if data:
            sizes, means, stds = zip(*data)
            if num_runs >= 30:
                score = norm.ppf((1 + confidence) / 2)
            else:
                score = t_dist.ppf((1 + confidence) / 2, num_runs - 1)
                
            plt.errorbar(sizes, means, yerr=score * np.array(stds) / np.sqrt(num_runs), marker='o', label=f'Gap {gap}')

    plt.title('Knapsack Problem: Execution time vs problem size for PuLP Gaps')
    plt.xlabel('Problem Size')
    plt.ylabel('Execution Time (s)')
    plt.grid(False)
    plt.legend()
    plt.xticks(sizes)
    plt.savefig('knapsack_execution_time_pulp_only.png')
    plt.close()

# Gráfico 2: Barras Agrupadas de Tempo (Padrão UC)
def plot_barras_tempo(results, sizes):
    x = np.arange(len(sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    tempos_guloso = [results['Guloso'][i][1] for i in range(len(sizes))]
    tempos_sa = [results['SA'][i][1] for i in range(len(sizes))]
    tempos_pulp = [results[0][i][1] for i in range(len(sizes))] # Gap 0

    rects1 = ax.bar(x - width, tempos_guloso, width, label='Guloso', color='#8c8c8c')
    rects2 = ax.bar(x, tempos_sa, width, label='Simulated Annealing', color='#e6a573')
    rects3 = ax.bar(x + width, tempos_pulp, width, label='PuLP (Ótimo)', color='#2d8c7c')

    ax.set_ylabel('Tempo Médio (segundos)')
    ax.set_xlabel('Tamanho do Problema (Número de Itens)')
    ax.set_title('Comparativo de Desempenho: Tempo Médio de Execução')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.grid(axis='y', linestyle='-', alpha=0.5)
    ax.legend()

    ax.bar_label(rects1, fmt='%.3f', padding=3, rotation=90, size=8)
    ax.bar_label(rects2, fmt='%.3f', padding=3, rotation=90, size=8)
    ax.bar_label(rects3, fmt='%.3f', padding=3, rotation=90, size=8)

    fig.tight_layout()
    plt.savefig('comparativo_barras_tempo.png')
    plt.close()

# Gráfico 3: Barras Agrupadas de Erro/Gap (Padrão UC)
def plot_barras_gap(sizes, gaps_guloso, gaps_sa):
    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width/2, gaps_guloso, width, label='Guloso', color='#8c8c8c')
    rects2 = ax.bar(x + width/2, gaps_sa, width, label='Simulated Annealing', color='#e6a573')
    # O PuLP não entra aqui pois o Gap dele para o ótimo é, por definição, 0%.

    ax.set_ylabel('Gap Médio em relação ao Ótimo (%)')
    ax.set_xlabel('Tamanho do Problema (Número de Itens)')
    ax.set_title('Comparativo de Qualidade: Gap Médio por Tamanho')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.grid(axis='y', linestyle='-', alpha=0.5)
    ax.legend()

    ax.bar_label(rects1, fmt='%.2f', padding=3, size=9)
    ax.bar_label(rects2, fmt='%.2f', padding=3, size=9)

    fig.tight_layout()
    plt.savefig('comparativo_barras_gap.png')
    plt.close()

def main():
    # ATENÇÃO: Tamanhos a partir de 100.000 podem demorar muitas horas para o PuLP Gap 0.
    sizes = [100, 1000, 10000, 100000, 1000000]
    max_weight = 50
    max_value = 100
    gap_thresholds = [0.4, 0.2, 0.1, 0.05, 0.02, 0]
    num_runs = 10
    
    results = {gap: [] for gap in gap_thresholds}
    results['Guloso'] = []
    results['SA'] = [] 
    
    lista_diferencas_guloso = [] 
    lista_diferencas_sa = []

    for size in sizes:
        items = generate_random_items(size, max_weight, max_value)
        capacity = size * max_weight / 2 

        print(f"\n[{size} ITENS] Rodando Guloso...")
        mean_time_g, std_time_g, obj_guloso = run_multiple_times(algoritmo_guloso, items, capacity, num_runs)
        results['Guloso'].append((size, mean_time_g, std_time_g))

        print(f"[{size} ITENS] Rodando Simulated Annealing...")
        mean_time_sa, std_time_sa, obj_sa = run_multiple_times(simulated_annealing, items, capacity, num_runs)
        results['SA'].append((size, mean_time_sa, std_time_sa))

        obj_otimo_pulp = 0

        for gap in gap_thresholds:
            print(f"[{size} ITENS] Rodando PuLP (Gap {gap})...")
            mean_time, std_time, obj_pulp = run_multiple_times(knapsack_problem_solver, items, capacity, num_runs, gap_threshold=gap)
            results[gap].append((size, mean_time, std_time))
            
            if gap == 0:
                obj_otimo_pulp = obj_pulp
        
        diferenca_guloso = ((obj_otimo_pulp - obj_guloso) / obj_otimo_pulp) * 100
        diferenca_sa = ((obj_otimo_pulp - obj_sa) / obj_otimo_pulp) * 100
        
        print(f"-> Gap Resultante: Guloso {diferenca_guloso:.2f}% | SA {diferenca_sa:.2f}%")
        
        lista_diferencas_guloso.append(diferenca_guloso)
        lista_diferencas_sa.append(diferenca_sa)

    # Gera os 3 gráficos solicitados
    plot_partial_results(results, sizes, num_runs)
    plot_barras_tempo(results, sizes)
    plot_barras_gap(sizes, lista_diferencas_guloso, lista_diferencas_sa)

    with open('knapsack_results.pickle', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()