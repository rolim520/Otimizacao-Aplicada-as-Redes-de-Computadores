import time as tm
import random
import math
import numpy as np
import matplotlib.pyplot as plt
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
    max_delta = 0.001
    peso_atual = 0
    valor_atual = 0
    mochila = [0] * len(items)
    
    amostras = min(500, len(items))
    for _ in range(amostras):
        indice, novo_bit, peso_viz, valor_viz = obter_movimento(mochila, items, capacity, peso_atual, valor_atual)
        delta = abs(valor_viz - valor_atual)
        
        if delta > max_delta:
            max_delta = delta
            
        if novo_bit == 1:
            mochila[indice] = 1; peso_atual = peso_viz; valor_atual = valor_viz
        else:
            mochila[indice] = 0; peso_atual = peso_viz; valor_atual = valor_viz
            
    t_inicial = max_delta / (-math.log(taxa_aceitacao_alvo))
    return t_inicial

def simulated_annealing(items, capacity, taxa_de_resfriamento=0.95, temp_final=0.1):
    start_time = tm.time()
    temperatura = calcular_temperatura_inicial(items, capacity)
    mochila_atual = [0] * len(items)
    peso_atual = 0
    valor_atual = 0
    melhor_valor = 0

    limite_sem_melhoria = len(items) * 2 

    while temperatura > temp_final:
        iteracoes_sem_melhoria = 0
        while iteracoes_sem_melhoria < limite_sem_melhoria:
            indice, novo_bit, peso_viz, valor_viz = obter_movimento(mochila_atual, items, capacity, peso_atual, valor_atual)
            delta = valor_viz - valor_atual

            aceitou = False
            if delta > 0:
                aceitou = True
                if valor_viz > melhor_valor:
                    melhor_valor = valor_viz
                    iteracoes_sem_melhoria = 0 
                else:
                    iteracoes_sem_melhoria += 1
            else:
                iteracoes_sem_melhoria += 1
                if random.random() < math.exp(delta / temperatura):
                    aceitou = True
            
            if aceitou:
                mochila_atual[indice] = novo_bit
                peso_atual = peso_viz
                valor_atual = valor_viz

        temperatura *= taxa_de_resfriamento 

    solve_time = tm.time() - start_time
    return melhor_valor, solve_time

# =====================================================================
# FUNÇÕES DE GERAÇÃO DE PROBLEMAS
# =====================================================================

def generate_correlated_items(num_items, max_weight, max_value):
    items = []
    for i in range(num_items):
        weight = random.randint(1, max_weight)
        value = weight + random.randint(1, 10) 
        items.append((i, weight, value))
    return items

def generate_uncorrelated_items(num_items, max_weight, max_value):
    items = []
    for i in range(num_items):
        weight = random.randint(1, max_weight)
        value = random.randint(1, max_value) 
        items.append((i, weight, value))
    return items

# =====================================================================
# PLOTAGEM DE GRÁFICOS E TABELAS
# =====================================================================

def plot_barras_tempo(results, sizes, prefix):
    """ Mantém o gráfico de barras dos tempos gerais por cenário """
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

def gerar_tabela_pulp_combinada(all_results, sizes, gap_thresholds):
    """ Gera uma imagem com a tabela de tempos do PuLP comparando os cenários """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    col_labels = ['Tamanho', 'Gap PuLP', 'Tempo Corr. (Média ± Std)', 'Tempo Desc. (Média ± Std)']
    table_data = []
    
    for size_idx, size in enumerate(sizes):
        for gap in gap_thresholds:
            row = [str(size), f"{gap}"]
            for cenario in ['Correlacionado', 'Descorrelacionado']:
                mean_t = all_results[cenario][gap][size_idx][1]
                std_t = all_results[cenario][gap][size_idx][2]
                row.append(f"{mean_t:.4f}s ± {std_t:.4f}s")
            table_data.append(row)
            
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8) # Ajusta espaçamento das células
    
    plt.title('Comparativo de Tempo do PuLP por Cenário e Gap', fontweight='bold', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig('Tabela_Comparativa_PuLP.png', bbox_inches='tight', dpi=300)
    plt.close()

def gerar_tabela_gap_combinada(all_gaps, sizes):
    """ Gera uma imagem com a tabela de Gaps comparando os cenários (Alta precisão) """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    col_labels = ['Tamanho', 'Gap Guloso (Corr)', 'Gap SA (Corr)', 'Gap Guloso (Desc)', 'Gap SA (Desc)']
    table_data = []
    
    for size_idx, size in enumerate(sizes):
        row = [str(size)]
        for cenario in ['Correlacionado', 'Descorrelacionado']:
            gap_g = all_gaps[cenario]['Guloso'][size_idx]
            gap_sa = all_gaps[cenario]['SA'][size_idx]
            
            # Formatação inteligente para mostrar precisão em números muito pequenos
            fmt_g = f"{gap_g:.6f}%" if gap_g > 1e-6 else f"{gap_g:.2e}%"
            fmt_sa = f"{gap_sa:.6f}%" if gap_sa > 1e-6 else f"{gap_sa:.2e}%"
            
            if gap_g == 0: fmt_g = "0.000000%"
            if gap_sa == 0: fmt_sa = "0.000000%"
            
            row.append(fmt_g)
            row.append(fmt_sa)
            
        table_data.append(row)
        
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    plt.title('Comparativo de Qualidade: Gap (%) em relação ao Ótimo', fontweight='bold', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig('Tabela_Comparativa_Gap.png', bbox_inches='tight', dpi=300)
    plt.close()

# =====================================================================
# MAIN LOOP
# =====================================================================

def main():
    random.seed(46)
    
    sizes = [100, 1000, 5000, 10000]
    max_weight = 10000
    max_value = 1000 
    gap_thresholds = [0.4, 0.1, 0.02, 0]
    num_runs = 5 
    
    cenarios = {
        "Correlacionado": generate_correlated_items,
        "Descorrelacionado": generate_uncorrelated_items
    }

    # Dicionários globais para guardar resultados de ambos os cenários
    all_results = {}
    all_gaps = {}

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
                items = func_geracao(size, max_weight, max_value)
                capacity = size * max_weight / 2 
                
                obj_otimo, t_otimo = knapsack_problem_solver(items, capacity, gap_threshold=0)
                tempos_run_pulp[0].append(t_otimo)

                for gap in gap_thresholds:
                    if gap != 0:
                        _, t_pulp = knapsack_problem_solver(items, capacity, gap_threshold=gap)
                        tempos_run_pulp[gap].append(t_pulp)

                obj_g, t_g = algoritmo_guloso(items, capacity)
                tempos_run_g.append(t_g)
                if obj_otimo > 0:
                    gaps_run_g.append(((obj_otimo - obj_g) / obj_otimo) * 100)
                else:
                    gaps_run_g.append(0)

                obj_sa, t_sa = simulated_annealing(items, capacity)
                tempos_run_sa.append(t_sa)
                if obj_otimo > 0:
                    gaps_run_sa.append(((obj_otimo - obj_sa) / obj_otimo) * 100)
                else:
                    gaps_run_sa.append(0)
                    
                print(f"    - Run {run+1}/{num_runs} concluída.")

            results['Guloso'].append((size, np.mean(tempos_run_g), np.std(tempos_run_g)))
            results['SA'].append((size, np.mean(tempos_run_sa), np.std(tempos_run_sa)))
            for gap in gap_thresholds:
                results[gap].append((size, np.mean(tempos_run_pulp[gap]), np.std(tempos_run_pulp[gap])))
            
            lista_diferencas_guloso.append(np.mean(gaps_run_g))
            lista_diferencas_sa.append(np.mean(gaps_run_sa))

        # Guarda no global para plotar depois
        all_results[nome_cenario] = results
        all_gaps[nome_cenario] = {
            'Guloso': lista_diferencas_guloso,
            'SA': lista_diferencas_sa
        }

        # Gráfico de barras normal ainda é útil por cenário
        plot_barras_tempo(results, sizes, nome_cenario)

    # =====================================================================
    # GERA AS TABELAS COMBINADAS NO FINAL
    # =====================================================================
    print("\nGerando tabelas consolidadas...")
    gerar_tabela_pulp_combinada(all_results, sizes, gap_thresholds)
    gerar_tabela_gap_combinada(all_gaps, sizes)
    
    with open('resultados_consolidados.pickle', 'wb') as f:
        pickle.dump({'resultados': all_results, 'gaps': all_gaps}, f)
        
    print("Concluído! Verifique as imagens: Tabela_Comparativa_PuLP.png e Tabela_Comparativa_Gap.png")

if __name__ == "__main__":
    main()