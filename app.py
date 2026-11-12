import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

st.set_page_config(
    page_title="Simulador de Dinâmica Inflacionária Termodinâmica",
    layout="wide"
)

st.title("Simulação Macroeconômica: Desequilíbrio Monetário-Produtivo")
st.markdown("""
*Proótipo computacional inspirado no modelo de Marinho & Rocha (2026)* para debate acadêmico. 
Este painel simula a inflação gerada pelo descompasso entre a manipulação de massa/energia produtiva 
ponderada pela carga de informação tecnológica e a expansão da massa monetária.
""")

# --- BARRA LATERAL: PARÂMETROS DE ENTRADA ---
st.sidebar.header("Parâmetros do Modelo")
num_cenarios = st.sidebar.slider("Número de Simulações (Monte Carlo)", 1000, 20000, 5000, 1000)
tempo_passos = st.sidebar.slider("Horizonte de Tempo (Meses)", 12, 120, 60, 12)

st.sidebar.subheader("Choques Estruturais e Volatilidade")
vol_moeda = st.sidebar.slider("Volatilidade da Massa Monetária (M)", 0.01, 0.08, 0.03, 0.005)
crescimento_produtivo = st.sidebar.slider("Crescimento Base da Produção (P)", 0.99, 1.02, 1.005, 0.001)
fator_tecnologia = st.sidebar.slider("Carga de Informação Tecnológica Média (I)", 0.8, 2.0, 1.2, 0.1)

# --- NÚCLEO DE SIMULAÇÃO (MONTE CARLO) ---
@st.cache_data
def rodar_simulacao(n_cenarios, n_tempo, v_moeda, cresc_prod, f_tec):
    inflacoes_medias = np.zeros(n_cenarios)
    trajetorias_exemplo = []
    
    for c in range(n_cenarios):
        # Setor produtivo real estocástico (Massa/Energia transformada)
        produtividade = np.random.normal(100, 4, n_tempo) * np.cumprod(np.random.normal(cresc_prod, 0.01, n_tempo))
        
        # Carga informacional / tecnológica local (ex: eficiência agregada)
        tecnologia_info = np.random.normal(f_tec, 0.05, n_tempo)
        
        # Massa monetária (M) injetada estocasticamente
        massa_monetaria = np.cumprod(np.random.normal(1.007, v_moeda, n_tempo)) * 100
        
        # Capacidade produtiva efetiva mapeada termodinamicamente: P * I
        capacidade_efetiva = produtividade * tecnologia_info
        
        # Desequilíbrio Monetário-Produtivo (D = M - P*I)
        desequilibrio = massa_monetaria - capacidade_efetiva
        
        # Formação de preços não-linear gerando a inflação local
        taxa_inflacao = np.maximum(0, desequilibrio / capacidade_efetiva) * np.random.uniform(0.4, 1.6, n_tempo)
        
        inflacoes_medias[c] = np.mean(taxa_inflacao)
        
        if c < 5: # Guardar algumas trajetórias para visualização
            trajetorias_exemplo.append(taxa_inflacao)
            
    return inflacoes_medias, trajetorias_exemplo

inflacoes_finais, amostras_trajetorias = rodar_simulacao(num_cenarios, tempo_passos, vol_moeda, crescimento_produtivo, fator_tecnologia)

# --- CÁLCULO DOS QUATRO MOMENTOS DA INFLAÇÃO ---
media_inf = np.mean(inflacoes_finais)
var_inf = np.var(inflacoes_finais)
skew_inf = skew(inflacoes_finais)
kurt_inf = kurtosis(inflacoes_finais) # Curtose excessiva

# --- EXIBIÇÃO DOS RESULTADOS NO DASHBOARD ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("1º Momento (Média)", f"{media_inf:.4f}")
col2.metric("2º Momento (Variância)", f"{var_inf:.5f}")
col3.metric("3º Momento (Assimetria)", f"{skew_inf:.3f}")
col4.metric("4º Momento (Curtose)", f"{kurt_inf:.3f}")

st.markdown("---")

# Gráficos de Análise
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Trajetórias de Inflação Simuladas (Exemplos)")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, traj in enumerate(amostras_trajetorias):
        ax.plot(traj, alpha=0.7, label=f"Cenário {i+1}")
    ax.set_title("Evolução Temporal da Inflação por Desequilíbrio")
    ax.set_xlabel("Tempo (Meses)")
    ax.set_ylabel("Taxa de Inflação Efetiva")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

with col_right:
    st.subheader("Distribuição dos Cenários de Inflação (Monte Carlo)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(inflacoes_finais, bins=50, density=True, color='purple', alpha=0.6, edgecolor='black')
    ax2.set_title("Densidade de Probabilidade das Médias de Inflação")
    ax2.set_xlabel("Taxa Média de Inflação")
    ax2.set_ylabel("Densidade")
    ax2.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig2)

st.markdown("---")
st.subheader("Pontos para Debate com o Autor:")
st.markdown(f"""
1. **Sensibilidade da Carga Informacional ($I$):** Ao ajustar o fator tecnológico na barra lateral, observe como a assimetria (3º momento: `{skew_inf:.2f}`) reage. Isso reflete a hipótese de que choques de produtividade informacional mitigam desvios inflacionários em economias emergentes?
2. **Não-linearidade do Desequilíbrio:** O modelo utiliza a função $\max(0, D / (P \cdot I))$. Em cenários de gargalos severos de cadeia de suprimentos (comuns no Brasil), essa não-linearidade gera caudas pesadas na curtose (`{kurt_inf:.2f}`). O autor enxerga essa formulação como suficiente para capturar choques de oferta exógenos?
""")