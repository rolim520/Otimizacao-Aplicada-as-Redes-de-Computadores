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

def obter_movimento(mochila, items, capacity, peso_atual, valor_atual):

    while True:
        indice = random.randint(0, len(mochila) - 1)
        
        if mochila[indice] == 1:
            # Informa: (índice alterado, novo bit, novo peso, novo valor)
            return indice, 0, peso_atual - items[indice][1], valor_atual - items[indice][2]
        else:
            if peso_atual + items[indice][1] <= capacity:
                return indice, 1, peso_atual + items[indice][1], valor_atual + items[indice][2]

def simulated_annealing(items, capacity, taxa_de_resfriamento=0.98, temp_inicial=1000, temp_final=1, max_iter=500):
    start_time = tm.time()
    mochila_atual = [0] * len(items)
    peso_atual = 0
    valor_atual = 0
    melhor_valor = 0

    temperatura = temp_inicial

    while temperatura > temp_final:
        for _ in range(max_iter):
            # Pega apenas a instrução do que mudar, sem copiar listas na memória!
            indice, novo_bit, peso_vizinho, valor_vizinho = obter_movimento(mochila_atual, items, capacity, peso_atual, valor_atual)
            
            delta = valor_vizinho - valor_atual

            # Variável de controle para saber se vamos efetivar a mudança na lista
            aceitou = False
            
            if delta > 0:
                aceitou = True
                if valor_vizinho > melhor_valor:
                    melhor_valor = valor_vizinho
            else:
                if random.random() < math.exp(delta / temperatura):
                    aceitou = True
            
            # Se a jogada foi aceita, nós alteramos O ÚNICO BIT na lista original!
            if aceitou:
                mochila_atual[indice] = novo_bit
                peso_atual = peso_vizinho
                valor_atual = valor_vizinho

        temperatura *= taxa_de_resfriamento

    solve_time = tm.time() - start_time
    return melhor_valor, solve_time

def generate_random_items(num_items, max_weight, max_value):
    items = []
    for i in range(num_items):
        weight = random.randint(1, max_weight)
        value = weight + random.randint(1, 10) 
        items.append((i, weight, value))
    return items

def run_multiple_times(solver_func, items, capacity, num_runs, **kwargs):
    times = []
    objetivos = []
    for i in range(num_runs):
        obj_val, solve_time = solver_func(items, capacity, **kwargs)
        times.append(solve_time)
        objetivos.append(obj_val)
        print("iteração", i)
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

# Gráfico 2: Barras Agrupadas de Tempo (Escala Log)
def plot_barras_tempo(results, sizes):
    x = np.arange(len(sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    tempos_guloso = [results['Guloso'][i][1] for i in range(len(sizes))]
    tempos_sa = [results['SA'][i][1] for i in range(len(sizes))]
    tempos_pulp = [results[0][i][1] for i in range(len(sizes))] 

    # Adicionando um valor ínfimo para não bugar o log(0)
    tempos_guloso = [max(t, 1e-4) for t in tempos_guloso] 

    rects1 = ax.bar(x - width, tempos_guloso, width, label='Guloso', color='#8c8c8c')
    rects2 = ax.bar(x, tempos_sa, width, label='Simulated Annealing', color='#e6a573')
    rects3 = ax.bar(x + width, tempos_pulp, width, label='PuLP (Ótimo)', color='#2d8c7c')

    # A MÁGICA: Escala logarítmica no eixo Y
    ax.set_yscale('log')
    
    ax.set_ylabel('Tempo Médio (segundos - Escala Log)')
    ax.set_xlabel('Tamanho do Problema (Número de Itens)')
    ax.set_title('Comparativo de Desempenho: Tempo Médio de Execução')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')

    ax.bar_label(rects1, fmt='%.3f', padding=3, rotation=90, size=8)
    ax.bar_label(rects2, fmt='%.3f', padding=3, rotation=90, size=8)
    ax.bar_label(rects3, fmt='%.3f', padding=3, rotation=90, size=8)

    # Aumentando a margem superior para o texto não cortar no topo
    ax.set_ylim(top=max(max(tempos_pulp), max(tempos_sa)) * 5) 

    fig.tight_layout()
    plt.savefig('comparativo_barras_tempo.png')
    plt.close()

# Gráfico 3: Barras Agrupadas de Erro/Gap (Escala Log)
def plot_barras_gap(sizes, gaps_guloso, gaps_sa):
    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Adicionando um valor ínfimo para não bugar o log(0)
    gaps_guloso_plot = [max(g, 1e-3) for g in gaps_guloso]
    gaps_sa_plot = [max(g, 1e-3) for g in gaps_sa]

    rects1 = ax.bar(x - width/2, gaps_guloso_plot, width, label='Guloso', color='#8c8c8c')
    rects2 = ax.bar(x + width/2, gaps_sa_plot, width, label='Simulated Annealing', color='#e6a573')

    # A MÁGICA: Escala logarítmica no eixo Y
    ax.set_yscale('log')

    ax.set_ylabel('Gap Médio em relação ao Ótimo (% - Escala Log)')
    ax.set_xlabel('Tamanho do Problema (Número de Itens)')
    ax.set_title('Comparativo de Qualidade: Gap Médio por Tamanho')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')

    # Para a anotação, usamos o valor original para ficar bonito no texto (ex: 0.00 ao invés de 0.001)
    ax.bar_label(rects1, labels=[f'{g:.2f}' for g in gaps_guloso], padding=3, size=9)
    ax.bar_label(rects2, labels=[f'{g:.2f}' for g in gaps_sa], padding=3, size=9)

    ax.set_ylim(bottom=1e-3, top=max(gaps_sa) * 5) 

    fig.tight_layout()
    plt.savefig('comparativo_barras_gap.png')
    plt.close()

def main():
    random.seed(46)
    # Tamanhos ideais para a apresentação
    sizes = [100, 1000, 10000, 100000]
    max_weight = 10000
    max_value = 100
    
    # Gaps reduzidos conforme solicitado
    gap_thresholds = [0.4, 0.1, 0.02, 0]
    
    # Alterado para apenas 5 execuções de média
    num_runs = 5 
    
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
        
        lista_diferencas_guloso.append(diferenca_guloso)
        lista_diferencas_sa.append(diferenca_sa)

    # =========================================================================
    # TABELA IMPRESSA NO TERMINAL
    # =========================================================================
    print("\n" + "="*75)
    print(f"RESUMO DOS RESULTADOS (MÉDIA DE {num_runs} EXECUÇÕES)")
    print("="*75)
    print(f"{'Tamanho':<10} | {'Método':<20} | {'Tempo Médio (s)':<15} | {'Gap (%)':<10}")
    print("-" * 75)

    for i, size in enumerate(sizes):
        # Pegando os tempos das listas
        tempo_pulp = results[0][i][1]
        tempo_guloso = results['Guloso'][i][1]
        tempo_sa = results['SA'][i][1]

        # Pegando os gaps
        gap_pulp = 0.00  # Por definição, a base do PuLP é 0
        gap_guloso = lista_diferencas_guloso[i]
        gap_sa = lista_diferencas_sa[i]

        print(f"{size:<10} | {'PuLP (Ótimo)':<20} | {tempo_pulp:<15.4f} | {gap_pulp:<10.2f}")
        print(f"{size:<10} | {'Guloso':<20} | {tempo_guloso:<15.4f} | {gap_guloso:<10.2f}")
        print(f"{size:<10} | {'Simulated Annealing':<20} | {tempo_sa:<15.4f} | {gap_sa:<10.2f}")
        print("-" * 75)
    # =========================================================================

    # Plota os gráficos com as funções que atualizamos
    plot_partial_results(results, sizes, num_runs)
    plot_barras_tempo(results, sizes)
    plot_barras_gap(sizes, lista_diferencas_guloso, lista_diferencas_sa)

    with open('knapsack_results.pickle', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
