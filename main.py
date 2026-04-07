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

# =====================================================================
# SOLVERS E HEURÍSTICAS
# =====================================================================

def knapsack_problem_solver(items, capacity, gap_threshold=0):
    problem = LpProblem("Knapsack_Problem", LpMaximize)
    x = LpVariable.dicts("Item", [i for i, _, _ in items], 0, 1, LpBinary)
    problem += lpSum([value * x[item] for item, _, value in items])
    problem += lpSum([weight * x[item] for item, weight, _ in items]) <= capacity

    if DEBUG_MODE:
        problem.writeLP("knapsack.lp")

    start_time = tm.time()
    # Adicionado timeLimit de 60 segundos para evitar travamentos em instâncias caóticas
    problem.solve(PULP_CBC_CMD(gapRel=gap_threshold, msg=False, timeLimit=60))
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
            return indice, 0, peso_atual - items[indice][1], valor_atual - items[indice][2]
        else:
            if peso_atual + items[indice][1] <= capacity:
                return indice, 1, peso_atual + items[indice][1], valor_atual + items[indice][2]

def calcular_temperatura_inicial(items, capacity, taxa_aceitacao_alvo=0.80):
    """
    Calcula a variação máxima de custo amostrando vizinhos e define uma 
    temperatura inicial onde a pior jogada tenha X% de chance de aceitação.
    """
    max_delta = 0.001
    peso_atual = 0
    valor_atual = 0
    mochila = [0] * len(items)
    
    # Amostra um número de vizinhos proporcional ao tamanho (limite max de 500)
    amostras = min(500, len(items))
    for _ in range(amostras):
        indice, novo_bit, peso_viz, valor_viz = obter_movimento(mochila, items, capacity, peso_atual, valor_atual)
        delta = abs(valor_viz - valor_atual)
        
        if delta > max_delta:
            max_delta = delta
            
        # Aceita aleatoriamente para passear pelo espaço
        if novo_bit == 1:
            mochila[indice] = 1; peso_atual = peso_viz; valor_atual = valor_viz
        else:
            mochila[indice] = 0; peso_atual = peso_viz; valor_atual = valor_viz
            
    # T = -Delta / ln(P)
    t_inicial = max_delta / (-math.log(taxa_aceitacao_alvo))
    return t_inicial

def simulated_annealing(items, capacity, taxa_de_resfriamento=0.95, temp_final=0.1):
    start_time = tm.time()
    
    # Calcula a temperatura inicial com base na dificuldade da instância
    temperatura = calcular_temperatura_inicial(items, capacity)
    
    mochila_atual = [0] * len(items)
    peso_atual = 0
    valor_atual = 0
    melhor_valor = 0

    # O limite de iterações sem melhorar a solução global é o dobro do tamanho da mochila
    limite_sem_melhoria = len(items) * 2 

    while temperatura > temp_final:
        iteracoes_sem_melhoria = 0
        
        # Loop Interno: Roda até estagnar as melhorias
        while iteracoes_sem_melhoria < limite_sem_melhoria:
            indice, novo_bit, peso_viz, valor_viz = obter_movimento(mochila_atual, items, capacity, peso_atual, valor_atual)
            delta = valor_viz - valor_atual

            aceitou = False
            
            if delta > 0: # Movimento de melhora
                aceitou = True
                if valor_viz > melhor_valor:
                    melhor_valor = valor_viz
                    iteracoes_sem_melhoria = 0 # Zerou! Achou uma solução melhor global
                else:
                    iteracoes_sem_melhoria += 1
            else: # Movimento de piora
                iteracoes_sem_melhoria += 1
                if random.random() < math.exp(delta / temperatura):
                    aceitou = True
            
            if aceitou:
                mochila_atual[indice] = novo_bit
                peso_atual = peso_viz
                valor_atual = valor_viz

        temperatura *= taxa_de_resfriamento # Resfria quando não consegue mais melhorar

    solve_time = tm.time() - start_time
    return melhor_valor, solve_time

# =====================================================================
# FUNÇÕES DE GERAÇÃO DE PROBLEMAS
# =====================================================================

def generate_correlated_items(num_items, max_weight, max_value):
    items = []
    for i in range(num_items):
        weight = random.randint(1, max_weight)
        # Valor atrelado ao peso (Fácil para Guloso)
        value = weight + random.randint(1, 10) 
        items.append((i, weight, value))
    return items

def generate_uncorrelated_items(num_items, max_weight, max_value):
    items = []
    for i in range(num_items):
        weight = random.randint(1, max_weight)
        # Valor totalmente caótico e independente (Difícil para Guloso)
        value = random.randint(1, max_value) 
        items.append((i, weight, value))
    return items

# =====================================================================
# PLOTAGEM DE GRÁFICOS
# =====================================================================

def plot_partial_results(results, sizes, num_runs, prefix, confidence=0.95):
    plt.figure(figsize=(10, 6))
    plt.xscale('log')
    for gap, data in results.items():
        if gap in ['Guloso', 'SA']: 
            continue
        if data:
            s_plot, means, stds = zip(*data)
            if num_runs >= 30:
                score = norm.ppf((1 + confidence) / 2)
            else:
                score = t_dist.ppf((1 + confidence) / 2, num_runs - 1)
            plt.errorbar(s_plot, means, yerr=score * np.array(stds) / np.sqrt(num_runs), marker='o', label=f'Gap {gap}')

    plt.title(f'{prefix} - Tempo de Execução vs Tamanho (PuLP)')
    plt.xlabel('Tamanho do Problema')
    plt.ylabel('Tempo (s)')
    plt.legend()
    plt.xticks(sizes, labels=sizes)
    plt.savefig(f'{prefix}_pulp_time.png')
    plt.close()

def plot_barras_tempo(results, sizes, prefix):
    x = np.arange(len(sizes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    tempos_guloso = [max(results['Guloso'][i][1], 1e-4) for i in range(len(sizes))]
    tempos_sa = [max(results['SA'][i][1], 1e-4) for i in range(len(sizes))]
    tempos_pulp = [max(results[0][i][1], 1e-4) for i in range(len(sizes))] 

    rects1 = ax.bar(x - width, tempos_guloso, width, label='Guloso', color='#8c8c8c')
    rects2 = ax.bar(x, tempos_sa, width, label='Simulated Annealing', color='#e6a573')
    rects3 = ax.bar(x + width, tempos_pulp, width, label='PuLP (Ótimo)', color='#2d8c7c')

    ax.set_yscale('log')
    ax.set_ylabel('Tempo Médio (segundos - Log)')
    ax.set_xlabel('Tamanho do Problema')
    ax.set_title(f'{prefix} - Tempo Médio de Execução')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left')

    ax.bar_label(rects1, fmt='%.3f', padding=3, rotation=90, size=8)
    ax.bar_label(rects2, fmt='%.3f', padding=3, rotation=90, size=8)
    ax.bar_label(rects3, fmt='%.3f', padding=3, rotation=90, size=8)

    ax.set_ylim(top=max(max(tempos_pulp), max(tempos_sa)) * 5) 
    fig.tight_layout()
    plt.savefig(f'{prefix}_barras_tempo.png')
    plt.close()

def plot_barras_gap(sizes, gaps_guloso, gaps_sa, prefix):
    x = np.arange(len(sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    gaps_guloso_plot = [max(g, 1e-3) for g in gaps_guloso]
    gaps_sa_plot = [max(g, 1e-3) for g in gaps_sa]

    rects1 = ax.bar(x - width/2, gaps_guloso_plot, width, label='Guloso', color='#8c8c8c')
    rects2 = ax.bar(x + width/2, gaps_sa_plot, width, label='Simulated Annealing', color='#e6a573')

    ax.set_yscale('log')
    ax.set_ylabel('Gap Médio em relação ao Ótimo (% - Log)')
    ax.set_xlabel('Tamanho do Problema')
    ax.set_title(f'{prefix} - Gap Médio por Tamanho')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left')

    ax.bar_label(rects1, labels=[f'{g:.2f}' for g in gaps_guloso], padding=3, size=9)
    ax.bar_label(rects2, labels=[f'{g:.2f}' for g in gaps_sa], padding=3, size=9)
    ax.set_ylim(bottom=1e-3, top=max(max(gaps_sa), max(gaps_guloso), 1) * 5) 

    fig.tight_layout()
    plt.savefig(f'{prefix}_barras_gap.png')
    plt.close()

# =====================================================================
# MAIN LOOP
# =====================================================================

def main():
    random.seed(46)
    
    # Reduzido para fechar em ~1h com segurança no num_runs = 5
    sizes = [100, 1000, 5000, 10000]
    max_weight = 10000
    max_value = 1000 # Escala de valor aumentada para o cenário caótico
    gap_thresholds = [0.4, 0.1, 0.02, 0]
    num_runs = 5 
    
    cenarios = {
        "Correlacionado": generate_correlated_items,
        "Descorrelacionado": generate_uncorrelated_items
    }

    for nome_cenario, func_geracao in cenarios.items():
        print(f"\n{'='*75}")
        print(f"INICIANDO BATERIA DE TESTES: CENÁRIO {nome_cenario.upper()}")
        print(f"{'='*75}")

        results = {gap: [] for gap in gap_thresholds}
        results['Guloso'] = []
        results['SA'] = [] 
        lista_diferencas_guloso = [] 
        lista_diferencas_sa = []

        for size in sizes:
            print(f"\n>>> Avaliando tamanho: {size} itens")
            
            tempos_run_g = []; gaps_run_g = []
            tempos_run_sa = []; gaps_run_sa = []
            tempos_run_pulp = {gap: [] for gap in gap_thresholds}
            
            for run in range(num_runs):
                # O SEGREDO: Nova mochila gerada para cada RUN!
                items = func_geracao(size, max_weight, max_value)
                capacity = size * max_weight / 2 
                
                # 1. Roda PuLP (Ótimo como base de cálculo)
                obj_otimo, t_otimo = knapsack_problem_solver(items, capacity, gap_threshold=0)
                tempos_run_pulp[0].append(t_otimo)

                # 2. Roda PuLP (Outros Gaps)
                for gap in gap_thresholds:
                    if gap != 0:
                        _, t_pulp = knapsack_problem_solver(items, capacity, gap_threshold=gap)
                        tempos_run_pulp[gap].append(t_pulp)

                # 3. Roda Guloso
                obj_g, t_g = algoritmo_guloso(items, capacity)
                tempos_run_g.append(t_g)
                # Proteção contra divisão por zero se obj_otimo for muito baixo ou 0
                if obj_otimo > 0:
                    gaps_run_g.append(((obj_otimo - obj_g) / obj_otimo) * 100)
                else:
                    gaps_run_g.append(0)

                # 4. Roda SA
                obj_sa, t_sa = simulated_annealing(items, capacity)
                tempos_run_sa.append(t_sa)
                if obj_otimo > 0:
                    gaps_run_sa.append(((obj_otimo - obj_sa) / obj_otimo) * 100)
                else:
                    gaps_run_sa.append(0)
                    
                print(f"    - Run {run+1}/{num_runs} concluída.")

            # Armazena as médias deste Tamanho
            results['Guloso'].append((size, np.mean(tempos_run_g), np.std(tempos_run_g)))
            results['SA'].append((size, np.mean(tempos_run_sa), np.std(tempos_run_sa)))
            for gap in gap_thresholds:
                results[gap].append((size, np.mean(tempos_run_pulp[gap]), np.std(tempos_run_pulp[gap])))
            
            lista_diferencas_guloso.append(np.mean(gaps_run_g))
            lista_diferencas_sa.append(np.mean(gaps_run_sa))

        # Tabela no Terminal para este Cenário
        print("\n" + "-"*75)
        print(f"RESUMO FINAL - {nome_cenario} (Média de {num_runs} execuções)")
        print("-" * 75)
        print(f"{'Tamanho':<10} | {'Método':<20} | {'Tempo Médio (s)':<15} | {'Gap (%)':<10}")
        print("-" * 75)
        
        for i, size in enumerate(sizes):
            tempo_pulp = results[0][i][1]
            tempo_guloso = results['Guloso'][i][1]
            tempo_sa = results['SA'][i][1]
            
            gap_guloso = lista_diferencas_guloso[i]
            gap_sa = lista_diferencas_sa[i]

            print(f"{size:<10} | {'PuLP (Ótimo)':<20} | {tempo_pulp:<15.4f} | {'0.00':<10}")
            print(f"{size:<10} | {'Guloso':<20} | {tempo_guloso:<15.4f} | {gap_guloso:<10.2f}")
            print(f"{size:<10} | {'Simulated Annealing':<20} | {tempo_sa:<15.4f} | {gap_sa:<10.2f}")
            print("-" * 75)

        # Gráficos exportados com prefixos
        plot_partial_results(results, sizes, num_runs, nome_cenario)
        plot_barras_tempo(results, sizes, nome_cenario)
        plot_barras_gap(sizes, lista_diferencas_guloso, lista_diferencas_sa, nome_cenario)

        with open(f'resultados_{nome_cenario.lower()}.pickle', 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    main()