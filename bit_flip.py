import matplotlib.pyplot as plt
import matplotlib.patches as patches

def criar_diagrama_vizinhanca():
    # Cores do Tema (convertidas para RGB hexadecimal padrão do matplotlib)
    COR_DESTAQUE_ESCURO = '#4ba173' # Verde escuro
    COR_FUNDO_CLARO = '#63d297'     # Verde claro
    COR_TEXTO_ESCURO = '#353744'    # Azul/Cinza escuro

    # Cria a figura
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off') # Remove os eixos do gráfico

    # Dados das mochilas (0 = Fora, 1 = Dentro)
    s_atual = [1, 0, 0, 1, 1]
    s_vizinho = [1, 0, 1, 1, 1]
    
    indice_alterado = 2 # É o 3º item (índice 2) que vamos alterar

    # Função auxiliar para desenhar a mochila com retângulos arredondados
    def desenhar_vetor(x_start, y_start, dados, titulo, destaque=None):
        # Título
        ax.text(x_start + 2.5, y_start + 1.4, titulo, 
                ha='center', va='center', fontsize=14, fontweight='bold', color=COR_TEXTO_ESCURO)
        
        # Desenha os quadrados arredondados e os números
        for i, bit in enumerate(dados):
            # Cores: Verde claro se está na mochila, Branco/Transparente se está fora
            cor_fundo = COR_FUNDO_CLARO if bit == 1 else 'white'
            
            # Se for o índice alterado, a borda fica mais grossa e com o verde escuro
            cor_borda = COR_DESTAQUE_ESCURO if i == destaque else COR_TEXTO_ESCURO
            espessura_borda = 3.5 if i == destaque else 1.5
            
            # Cria o retângulo com cantos arredondados
            # pad e rounding_size controlam a suavidade das bordas
            rect = patches.FancyBboxPatch((x_start + i + 0.1, y_start + 0.1), 0.8, 0.8, 
                                          boxstyle="round,pad=0.05,rounding_size=0.15",
                                          linewidth=espessura_borda, edgecolor=cor_borda, facecolor=cor_fundo)
            ax.add_patch(rect)
            
            # Escreve o bit (0 ou 1) no centro do quadrado
            cor_fonte = COR_TEXTO_ESCURO if bit == 0 else 'white' # Contraste para o texto interno
            if bit == 1 and i == destaque: cor_fonte = COR_TEXTO_ESCURO # Destaque visual no alterado
            
            ax.text(x_start + i + 0.5, y_start + 0.5, str(bit), 
                    ha='center', va='center', fontsize=22, fontweight='bold', color=cor_fonte)
            
            # Escreve o índice em cima da caixa
            ax.text(x_start + i + 0.5, y_start + 1.1, f'Item {i}', 
                    ha='center', va='center', fontsize=10, color=COR_TEXTO_ESCURO, alpha=0.8)

    # 1. Desenha a solução atual na esquerda
    desenhar_vetor(0, 2, s_atual, "Solução Atual (S)", destaque=indice_alterado)

    # 2. Desenha uma seta no meio indicando a ação
    ax.annotate("", xy=(6.5, 2.5), xytext=(5.5, 2.5), 
                arrowprops=dict(facecolor=COR_DESTAQUE_ESCURO, edgecolor='none', shrink=0.05, width=4, headwidth=12))
    ax.text(6.0, 2.8, "Sorteia um\níndice", ha='center', va='center', fontsize=11, fontweight='bold', color=COR_TEXTO_ESCURO)

    # 3. Desenha a solução vizinha na direita
    desenhar_vetor(7, 2, s_vizinho, "Vizinho Gerado (S')", destaque=indice_alterado)

    # 4. Adiciona texto explicativo da Inversão de Bit apontando para o alterado
    ax.annotate("Inversão de Bit\n(Bit Flip)", xy=(9.5, 1.9), xytext=(9.5, 0.8), 
                arrowprops=dict(facecolor=COR_DESTAQUE_ESCURO, edgecolor='none', shrink=0.05, width=2, headwidth=8),
                ha='center', va='top', fontsize=12, color=COR_DESTAQUE_ESCURO, fontweight='bold')

    # 5. Legenda geral arredondada usando a paleta
    ax.text(2.5, 1.3, "Legenda: [ 1 ] = Na Mochila  |  [ 0 ] = Fora da Mochila", 
            ha='center', va='top', fontsize=11, color=COR_TEXTO_ESCURO,
            bbox=dict(facecolor='white', edgecolor=COR_TEXTO_ESCURO, boxstyle='round,pad=0.5'))

    # Ajusta os limites da imagem e salva
    plt.xlim(-1, 13)
    plt.ylim(0.5, 4.5)
    plt.tight_layout()
    plt.savefig('vizinhanca_sa_tema.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Imagem 'vizinhanca_sa_tema.png' gerada com sucesso!")

if __name__ == "__main__":
    criar_diagrama_vizinhanca()