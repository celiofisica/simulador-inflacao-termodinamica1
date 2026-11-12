# -*- coding: utf-8 -*-
"""
Laboratório de Econofísica: inflação como desequilíbrio monetário-produtivo.

App didático baseado em
    Marinho, A. & Rocha, H. (2026). Analyses of Inflation Dynamics as a
    Consequence of Monetary-Production Imbalance in Global Markets.
    Computational Economics. https://doi.org/10.1007/s10614-026-11472-y

O que é do artigo e o que é deste app
-------------------------------------
* Abas 2 e 3 implementam diretamente as equações (2.1)-(2.6), (3.2)-(3.5),
  (4.1)-(4.8) e a classificação de regimes da seção 6.2.
* Aba 4 calcula os índices H, F, K, G (eqs. 4.13-4.16) sobre trajetórias
  sintéticas. O mecanismo de repasse para os preços (defasagem + pass-through)
  é uma HIPÓTESE DIDÁTICA deste app: o artigo apenas constata o atraso.
* Aba 5 aplica as mesmas equações a dados reais enviados pelo usuário.

Execução local:  streamlit run app.py
"""
from __future__ import annotations

import inspect
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Configuração e constantes
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Econofísica: inflação por desequilíbrio monetário-produtivo",
    page_icon="📈",
    layout="wide",
)

DOI_URL = "https://doi.org/10.1007/s10614-026-11472-y"
AGREGADOS = ["M0", "M1", "M2", "M3"]
INDICES = ["H", "F", "K", "G"]          # H<-M0, F<-M1, K<-M2, G<-M3
COR = {"H": "#1f77b4", "F": "#2ca02c", "K": "#ff7f0e", "G": "#d62728", "pi": "#9467bd"}
DEFASAGEM_MAX = 5                        # anos de memória do repasse (aba 4)
EPS_REGIME = 0.005                       # 0,5 p.p.: o que conta como "aproximadamente zero"


# ----------------------------------------------------------------------------
# Utilidades de interface (compatíveis com versões antigas e novas do Streamlit)
# ----------------------------------------------------------------------------
def _aceita_width(func) -> bool:
    try:
        return "width" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def mostrar(fig: go.Figure, altura: int = 420) -> None:
    """Exibe figura Plotly ocupando a largura disponível."""
    fig.update_layout(
        height=altura,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        hovermode="x unified",
    )
    if _aceita_width(st.plotly_chart):
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def tabela(df: pd.DataFrame) -> None:
    if _aceita_width(st.dataframe):
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def pct(x: float, casas: int = 2) -> str:
    """Formata fração como porcentagem no padrão brasileiro."""
    if x is None or not np.isfinite(x):
        return "n/d"
    return f"{100 * x:.{casas}f}".replace(".", ",") + " %"


def num(x: float, casas: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "n/d"
    return f"{x:.{casas}f}".replace(".", ",")


# ----------------------------------------------------------------------------
# Núcleo numérico
# ----------------------------------------------------------------------------
def momentos_estatisticos(x) -> tuple[float, float, float, float]:
    """Média, variância, assimetria e curtose excedente de uma amostra."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (np.nan,) * 4
    m, s = x.mean(), x.std()
    if s == 0:
        return m, 0.0, 0.0, 0.0
    z = (x - m) / s
    return m, s**2, float((z**3).mean()), float((z**4).mean() - 3.0)


def pesos_defasagem(tau: float) -> np.ndarray:
    """Pesos de Poisson (média tau) truncados em 0..DEFASAGEM_MAX e normalizados."""
    h = np.arange(DEFASAGEM_MAX + 1)
    if tau <= 0:
        w = np.zeros(DEFASAGEM_MAX + 1)
        w[0] = 1.0
        return w
    w = np.array([math.exp(-tau) * tau**k / math.factorial(k) for k in h])
    return w / w.sum()


def classificar_regime(ds, dv, eps: float = EPS_REGIME) -> np.ndarray:
    """Regimes da seção 6.2 (com dois rótulos extras para cobrir o plano todo).

    ds = variação relativa de sigma_T (produção); dv = variação relativa de v.
    """
    ds, dv = np.asarray(ds, dtype=float), np.asarray(dv, dtype=float)
    r = np.full(ds.shape, "Estagnado", dtype=object)
    r[(np.abs(dv) <= eps) & (ds > eps)] = "Crescimento real / equilibrado"
    r[(dv > eps) & (ds > eps)] = "Misto: moeda cresce mais que a produção"
    r[(dv > eps) & (np.abs(ds) <= eps)] = "Crescimento financeiro"
    r[(dv > eps) & (ds < -eps)] = "Inflacionário"
    r[dv < -eps] = "v em queda: moeda cresce menos que a produção"
    return r


@st.cache_data(show_spinner=False)
def gerar_bens(n_bens: int, semente: int):
    """Economia de brinquedo: quantidades, custo material e |ΔS| de cada bem."""
    rng = np.random.default_rng(semente)
    quantidade = rng.integers(10, 1000, n_bens).astype(float)     # N_j
    v_material = rng.lognormal(mean=1.0, sigma=0.6, size=n_bens)  # V_R = Σ b_k m_k
    delta_s = rng.gamma(shape=2.0, scale=1.0, size=n_bens)        # |ΔS_j|
    ruido = rng.standard_normal(n_bens)                           # forma de D_j - S_j
    return quantidade, v_material, delta_s, ruido


@st.cache_data(show_spinner=False, max_entries=4)
def simular(n: int, anos: int, semente: int, g: float, sig_g: float, mu0: float,
            spread: float, sig_m: float, rho: float, ano_choque: int,
            choque_m: float, choque_y: float, k_motor: int, theta: float,
            tau: float, sig_pi: float) -> dict:
    """Monte Carlo vetorizado. Todas as taxas são frações anuais (0,02 = 2 %).

    Produção:   Y_{t+1} = Y_t (1 + g + ruído - choque)
    Moeda:      Mk_{t+1} = Mk_t (1 + mu0 + k*spread + ruído comum + ruído próprio + choque)
    Índices:    I_k = (Mk_{t+1}/Y_{t+1}) / (Mk_t/Y_t) - 1          [eqs. 4.13-4.16]
    Preços:     pi_t = theta * Σ_h w_h I_motor(t-h) + ruído        [hipótese do app]
    """
    rng = np.random.default_rng(semente)
    T = anos + DEFASAGEM_MAX                       # anos extras de aquecimento
    gy = g + sig_g * rng.standard_normal((n, T))
    comum = rng.standard_normal((n, T))
    gm = np.empty((4, n, T))
    for k in range(4):
        proprio = rng.standard_normal((n, T))
        gm[k] = (mu0 + k * spread) + sig_m * (math.sqrt(rho) * comum + math.sqrt(1 - rho) * proprio)
    if ano_choque > 0:
        j = DEFASAGEM_MAX + ano_choque - 1
        gm[:, :, j] += choque_m
        gy[:, j] -= choque_y
    gy = np.clip(gy, -0.9, None)                   # impede produção/moeda negativas
    gm = np.clip(gm, -0.9, None)

    indices = (1.0 + gm) / (1.0 + gy) - 1.0        # forma exata, sem linearizar
    motor = indices[k_motor]
    pressao = np.zeros((n, T))
    for h, w in enumerate(pesos_defasagem(tau)):
        if w > 0:
            pressao[:, h:] += w * motor[:, : T - h]
    inflacao = theta * pressao + sig_pi * rng.standard_normal((n, T))

    corte = slice(DEFASAGEM_MAX, None)
    return {"I": indices[:, :, corte], "pi": inflacao[:, corte],
            "gy": gy[:, corte], "gm": gm[:, :, corte]}


def correlacao_defasada(indice: np.ndarray, inflacao: np.ndarray, h_max: int) -> np.ndarray:
    """rho(h) = Corr(I_t, pi_{t+h}) por cenário (eq. 5.1). Retorna matriz (h_max+1, n)."""
    T = indice.shape[1]
    saida = []
    for h in range(h_max + 1):
        a, b = indice[:, : T - h], inflacao[:, h:]
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            saida.append(np.where(den > 0, (a * b).sum(axis=1) / den, np.nan))
    return np.array(saida)


def mqo(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Mínimos quadrados com intercepto. Retorna coeficientes e R²."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuo = y - X @ beta
    sq_total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residuo**2).sum()) / sq_total if sq_total > 0 else np.nan
    return beta, r2


# ----------------------------------------------------------------------------
# Leitura robusta de dados do usuário (aba 5)
# ----------------------------------------------------------------------------
APELIDOS = {
    "ano": ["ano", "year", "data", "date", "t"],
    "PIB": ["pib", "gdp", "produto", "sigma_t"],
    "inflacao": ["inflacao", "inflação", "inflation", "ipca", "cpi", "pi"],
    "M0": ["m0"], "M1": ["m1"], "M2": ["m2"], "M3": ["m3"],
}


def _para_numero(serie: pd.Series) -> pd.Series:
    """Aceita '1.234,56', '1234.56', '5,2%' etc."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    s = serie.astype(str).str.strip().str.replace("%", "", regex=False).str.replace(" ", "", regex=False)
    tem_virgula = s.str.contains(",", regex=False)
    s = s.where(~tem_virgula, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def ler_csv(conteudo: bytes) -> tuple[pd.DataFrame | None, list[str]]:
    """Lê CSV com separador , ; ou tab e decimal com ponto ou vírgula."""
    avisos: list[str] = []
    try:
        texto = conteudo.decode("utf-8-sig", errors="replace")
        bruto = pd.read_csv(io.StringIO(texto), sep=None, engine="python")
    except Exception as erro:  # noqa: BLE001 - queremos mostrar qualquer falha ao usuário
        return None, [f"Não consegui ler o arquivo: {erro}"]

    mapa = {}
    for coluna in bruto.columns:
        chave = str(coluna).strip().lower()
        for padrao, nomes in APELIDOS.items():
            if chave in nomes and padrao not in mapa.values():
                mapa[coluna] = padrao
    df = bruto.rename(columns=mapa)[list(mapa.values())].copy()

    if "ano" not in df or "PIB" not in df:
        return None, ["O arquivo precisa ter, no mínimo, as colunas 'ano' e 'PIB'."]
    if not any(m in df for m in AGREGADOS):
        return None, ["Inclua pelo menos um agregado monetário: M0, M1, M2 ou M3."]

    for coluna in df.columns:
        df[coluna] = _para_numero(df[coluna])
    antes = len(df)
    df = df.dropna(subset=["ano", "PIB"]).drop_duplicates("ano").sort_values("ano")
    if np.allclose(df["ano"] % 1, 0):
        df["ano"] = df["ano"].astype(int)
    if len(df) < antes:
        avisos.append(f"{antes - len(df)} linha(s) descartada(s) por ano/PIB ausente ou ano repetido.")
    positivos = [c for c in ["PIB"] + AGREGADOS if c in df]
    if (df[positivos] <= 0).any().any():
        avisos.append("Há valores nulos ou negativos em PIB/agregados; essas linhas foram removidas.")
        df = df[(df[positivos] > 0).all(axis=1)]
    if len(df) < 4:
        return None, avisos + ["Sobraram menos de 4 anos válidos; não dá para analisar."]
    if (np.diff(df["ano"].to_numpy()) != 1).any():
        avisos.append("Os anos não são consecutivos: as variações cobrem intervalos diferentes.")
    return df.reset_index(drop=True), avisos


def calcular_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Índices em % a.a., datados no ano de chegada (variação de t-1 para t)."""
    saida = pd.DataFrame({"ano": df["ano"].to_numpy()[1:]})
    for agregado, nome in zip(AGREGADOS, INDICES):
        if agregado in df:
            v = (df[agregado] / df["PIB"]).to_numpy()
            saida[nome] = 100.0 * (v[1:] / v[:-1] - 1.0)
    if "inflacao" in df:
        saida["inflacao"] = df["inflacao"].to_numpy()[1:]
    return saida


@st.cache_data(show_spinner=False)
def exemplo_sintetico() -> pd.DataFrame:
    """Um cenário do simulador convertido em níveis. NÃO são dados reais."""
    anos = 30
    s = simular(1, anos, 7, 0.015, 0.015, 0.035, 0.01, 0.015, 0.7, 26, 0.05, 0.04, 2, 0.6, 2.0, 0.005)
    pib = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + s["gy"][0])])
    dados = {"ano": np.arange(1995, 1995 + anos + 1), "PIB": pib.round(3)}
    for k, (agregado, base) in enumerate(zip(AGREGADOS, [10.0, 25.0, 60.0, 80.0])):
        dados[agregado] = (base * np.concatenate([[1.0], np.cumprod(1.0 + s["gm"][k, 0])])).round(3)
    dados["inflacao"] = np.concatenate([[np.nan], 100.0 * s["pi"][0]]).round(2)
    return pd.DataFrame(dados)


# ----------------------------------------------------------------------------
# Cenários prontos da aba 4 (caricaturas didáticas, não calibrações)
# ----------------------------------------------------------------------------
PRESETS = {
    "Economia estável (tipo Alemanha)": dict(
        p_g=1.5, p_sig_g=1.5, p_mu0=3.5, p_spread=1.0, p_sig_m=1.5,
        p_choque_m=5.0, p_choque_y=4.0, p_theta=0.6, p_tau=2.0, p_sig_pi=0.5),
    "Emergente (tipo Brasil)": dict(
        p_g=2.5, p_sig_g=2.5, p_mu0=7.0, p_spread=1.5, p_sig_m=3.0,
        p_choque_m=6.0, p_choque_y=4.0, p_theta=0.8, p_tau=1.0, p_sig_pi=1.5),
    "Transmissão bloqueada (tipo Japão/China)": dict(
        p_g=1.0, p_sig_g=1.5, p_mu0=4.0, p_spread=1.0, p_sig_m=1.5,
        p_choque_m=5.0, p_choque_y=4.0, p_theta=0.05, p_tau=2.0, p_sig_pi=0.6),
    "Hiperinflação (tipo Venezuela)": dict(
        p_g=-2.0, p_sig_g=5.0, p_mu0=60.0, p_spread=5.0, p_sig_m=25.0,
        p_choque_m=150.0, p_choque_y=10.0, p_theta=1.3, p_tau=0.5, p_sig_pi=10.0),
}
PRESET_PADRAO = next(iter(PRESETS))
for _chave, _valor in PRESETS[PRESET_PADRAO].items():
    st.session_state.setdefault(_chave, _valor)


def aplicar_preset() -> None:
    for chave, valor in PRESETS[st.session_state["preset"]].items():
        st.session_state[chave] = valor


# ----------------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------------
st.title("Laboratório de Econofísica")
st.subheader("Inflação como desequilíbrio entre moeda e produção")
st.markdown(
    f"Material didático interativo baseado em **Marinho & Rocha (2026)**, "
    f"*Computational Economics* ([doi:10.1007/s10614-026-11472-y]({DOI_URL})). "
    "É uma leitura independente para sala de aula e debate: não é implementação oficial dos autores."
)

aba_ideia, aba_preco, aba_fiat, aba_mc, aba_dados, aba_debate = st.tabs([
    "1 · A ideia", "2 · Preço e o parâmetro v", "3 · Inflação fiat",
    "4 · Monte Carlo: H, F, K, G", "5 · Seus dados", "6 · Questões em aberto",
])

# ============================================================================
# ABA 1 - A IDEIA
# ============================================================================
with aba_ideia:
    esquerda, direita = st.columns([3, 2])
    with esquerda:
        st.markdown("### O argumento em cinco passos")
        st.markdown(
            "**1. Valor intrínseco.** Produzir um bem é reorganizar matéria. O custo tem uma parte "
            "material (as massas $m_k$ das substâncias, a preço $b_k$) e uma parte informacional "
            "(a variação de entropia $|\\Delta S_j|$ imposta à matéria, a preço $b_A$):"
        )
        st.latex(r"[\sigma_j]_{\rm net}=\underbrace{\sum_k b_k\,(m_k)_j}_{\mathcal V_R}"
                 r"+\underbrace{b_A\,|\Delta S_j|}_{\mathcal V_A}")
        st.markdown("**2. Mercado.** O preço observado se afasta do intrínseco conforme o excesso "
                    "de demanda $D_j-S_j$, com sensibilidade $c$:")
        st.latex(r"[\sigma_j]_{\rm market}=e^{\,c\,(D_j-S_j)}\,[\sigma_j]_{\rm net}")
        st.markdown("**3. Agregação.** Somando sobre todos os bens (quantidades $N_j$) obtém-se o valor "
                    "intrínseco total $\\sigma_T$, o valor nominal $\\langle\\mathcal P\\rangle$ e a razão $v$:")
        st.latex(r"\sigma_T=\sum_j N_j[\sigma_j]_{\rm net},\qquad"
                 r"\langle\mathcal P\rangle=\sum_j N_j[\sigma_j]_{\rm market},\qquad"
                 r"v=\frac{\langle\mathcal P\rangle}{\sigma_T}")
        st.markdown("**4. Inflação fiat.** A inflação é *definida* como a taxa de variação de $v$:")
        st.latex(r"i_f=\frac{v_{t+1}}{v_t}-1\;\approx\;"
                 r"\frac{\delta\langle\mathcal P\rangle}{\langle\mathcal P\rangle}"
                 r"-\frac{\delta\sigma_T}{\sigma_T}")
        st.markdown("**5. Do modelo aos dados.** Troca-se $\\sigma_T$ pelo PIB e "
                    "$\\langle\\mathcal P\\rangle$ por um agregado monetário. Cada agregado gera um índice:")
        st.latex(r"\Pi^{(M)}_t=\frac{M_{t+1}/\mathrm{PIB}_{t+1}}{M_t/\mathrm{PIB}_t}-1"
                 r"\qquad\Rightarrow\qquad H\,(M0),\;F\,(M1),\;K\,(M2),\;G\,(M3)")
    with direita:
        st.markdown("### Como usar este app")
        st.markdown(
            "- **Aba 2:** monte uma economia de brinquedo e veja $v$ nascer dos preços.\n"
            "- **Aba 3:** faça $v$ evoluir no tempo e teste a aproximação linear.\n"
            "- **Aba 4:** simule milhares de economias, calcule H, F, K, G e estude o atraso "
            "entre os índices e a inflação.\n"
            "- **Aba 5:** aplique as fórmulas a dados reais.\n"
            "- **Aba 6:** questões em aberto e extensões possíveis."
        )
        st.warning(
            "**Cuidado com a palavra “momento”.** No artigo, os quatro *momentos de inflação* são "
            "os índices H, F, K e G, um para cada agregado monetário. Eles **não** são média, "
            "variância, assimetria e curtose. Os momentos estatísticos aparecem na aba 4 apenas "
            "para descrever a dispersão entre cenários simulados."
        )
        with st.expander("Glossário rápido"):
            st.markdown(
                "- **M0:** base monetária (papel-moeda e reservas).\n"
                "- **M1:** M0 em poder do público + depósitos à vista.\n"
                "- **M2:** M1 + poupança e depósitos a prazo.\n"
                "- **M3:** M2 + instrumentos menos líquidos (fundos, compromissadas).\n"
                "- **$v$:** quanto valor nominal existe por unidade de valor produzido.\n"
                "- **Repasse (*pass-through*):** fração da pressão monetária que chega aos preços ao consumidor."
            )

# ============================================================================
# ABA 2 - PREÇO E O PARÂMETRO v
# ============================================================================
with aba_preco:
    st.markdown("### Uma economia de brinquedo, bem por bem")
    st.caption("Implementa as eqs. (2.1) a (2.6) e (3.2) a (3.5). Cada círculo do gráfico é um tipo de bem.")
    controles, graficos = st.columns([1, 3])
    with controles:
        n_bens = st.slider("Número de tipos de bens", 5, 60, 30, 5)
        c_sens = st.slider("Sensibilidade do preço, c", 0.0, 2.0, 0.5, 0.05,
                           help="Quanto o preço reage ao excesso de demanda. c = 0 desliga o mercado.")
        media_x = st.slider("Excesso de demanda médio ⟨D − S⟩", -1.0, 1.0, 0.0, 0.05,
                            help="Média ponderada pelo valor de cada bem na economia.")
        disp_x = st.slider("Dispersão de D − S entre os bens", 0.0, 2.0, 0.0, 0.05,
                           help="Desvio-padrão ponderado. Zero = todos os bens com o mesmo desequilíbrio.")
        b_a = st.slider("Preço da informação, b_A", 0.0, 5.0, 1.0, 0.1,
                        help="Valor monetário por unidade de |ΔS|. Mede o peso da tecnologia no valor.")
        semente_bens = st.number_input("Semente da economia", 0, 9999, 42, 1, key="semente_bens")

    quantidade, v_material, delta_s, ruido = gerar_bens(n_bens, int(semente_bens))
    v_info = b_a * delta_s
    sigma_net = v_material + v_info
    peso = quantidade * sigma_net / (quantidade * sigma_net).sum()
    media_r = (peso * ruido).sum()
    desvio_r = math.sqrt((peso * (ruido - media_r) ** 2).sum())
    excesso = media_x + disp_x * (ruido - media_r) / desvio_r      # média e desvio ponderados exatos
    multiplicador = np.exp(c_sens * excesso)
    sigma_mercado = multiplicador * sigma_net
    sigma_t = (quantidade * sigma_net).sum()
    p_nominal = (quantidade * sigma_mercado).sum()
    v_micro = p_nominal / sigma_t
    v_ordem1 = 1.0 + c_sens * media_x
    v_ordem2 = v_ordem1 + 0.5 * c_sens**2 * (media_x**2 + disp_x**2)

    with graficos:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("σ_T (intrínseco)", num(sigma_t, 0))
        m2.metric("⟨P⟩ (nominal)", num(p_nominal, 0))
        m3.metric("v = ⟨P⟩/σ_T", num(v_micro, 4), delta=pct(v_micro - 1.0), delta_color="inverse")
        m4.metric("Parcela informacional de σ_T", pct((quantidade * v_info).sum() / sigma_t, 1),
                  help="Fração do valor intrínseco que vem de b_A·|ΔS|.")

        grade = np.linspace(min(excesso.min(), -0.5) - 0.2, max(excesso.max(), 0.5) + 0.2, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grade, y=np.exp(c_sens * grade), mode="lines",
                                 name="e^{c(D−S)}", line=dict(color="gray", dash="dot")))
        fig.add_trace(go.Scatter(
            x=excesso, y=multiplicador, mode="markers", name="bens (área ∝ peso)",
            marker=dict(size=8 + 60 * np.sqrt(peso), color=COR["K"], opacity=0.7,
                        line=dict(width=1, color="white"))))
        fig.add_hline(y=v_micro, line_color=COR["G"],
                      annotation_text=f"v = {num(v_micro, 3)}", annotation_position="top left")
        fig.add_hline(y=1.0, line_color="gray", line_dash="dash")
        fig.update_layout(title="Multiplicador de preço de cada bem",
                          xaxis_title="excesso de demanda D − S",
                          yaxis_title="preço de mercado / valor intrínseco", hovermode="closest")
        mostrar(fig, 400)

        st.markdown(
            f"**Conferindo a conta.** Exato: v = {num(v_micro, 4)}. "
            f"Expansão de 1ª ordem, 1 + c⟨D−S⟩ = {num(v_ordem1, 4)}. "
            f"Até 2ª ordem, somando c²⟨(D−S)²⟩/2: {num(v_ordem2, 4)}."
        )

    with st.expander("🧪 Experimentos sugeridos (tente prever antes de mexer)"):
        st.markdown(
            "1. **Equilíbrio total.** Deixe média e dispersão em zero. Resultado: v = 1, que é a eq. (2.5).\n"
            "2. **Equilíbrio só na média.** Mantenha a média em zero e aumente a dispersão. O artigo "
            "afirma (eq. 3.5) que basta ⟨D−S⟩ = 0 para v = 1. O que acontece com v?\n"
            "3. **Tecnologia.** Aumente b_A. A parcela informacional cresce, mas v quase não muda. Por quê?"
        )
    with st.expander("Respostas"):
        st.markdown(
            "2. v fica **maior** que 1. A exponencial é convexa, então pela desigualdade de Jensen "
            "$\\langle e^{cx}\\rangle \\ge e^{c\\langle x\\rangle}$. A eq. (3.5) só vale em primeira "
            "ordem em $c$; o termo seguinte é $c^2\\langle x^2\\rangle/2 > 0$. Consequência física: "
            "mercados mais *heterogêneos* inflam $v$ mesmo com oferta e demanda equilibradas na média.\n\n"
            "3. $v$ é uma razão: $b_A$ entra no numerador e no denominador. A termodinâmica define o "
            "*nível* de $\\sigma_T$, mas $v$ só enxerga o desvio do mercado em relação a esse nível. "
            "Guarde isso para a aba 6."
        )

# ============================================================================
# ABA 3 - INFLAÇÃO FIAT
# ============================================================================
with aba_fiat:
    st.markdown("### De v(t) à inflação fiat")
    st.caption("Implementa as eqs. (4.1) a (4.8) e a classificação de regimes da seção 6.2.")
    controles, graficos = st.columns([1, 3])
    with controles:
        mu_det = st.slider("Crescimento do valor nominal ⟨P⟩ (% a.a.)", -10.0, 500.0, 8.0, 0.5) / 100
        g_det = st.slider("Crescimento da produção σ_T (% a.a.)", -10.0, 15.0, 2.0, 0.5) / 100
        anos_det = st.slider("Horizonte (anos)", 5, 50, 20, 5)
        escala_log = st.checkbox("Eixo vertical logarítmico", value=mu_det > 0.5)

    fiat_exato = (1 + mu_det) / (1 + g_det) - 1
    fiat_linear = mu_det - g_det
    t = np.arange(anos_det + 1)
    serie_p, serie_s = 100 * (1 + mu_det) ** t, 100 * (1 + g_det) ** t
    serie_v = serie_p / serie_s

    with graficos:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("i_f exato", pct(fiat_exato), help="(1+μ)/(1+g) − 1, eq. (4.7)")
        m2.metric("i_f linearizado", pct(fiat_linear), help="μ − g, eq. (4.8)")
        m3.metric("Erro da linearização", pct(fiat_linear - fiat_exato))
        m4.metric("Regime (seção 6.2)", str(classificar_regime(g_det, fiat_exato)).split(":")[0])
        st.caption(f"Regime completo: {classificar_regime(g_det, fiat_exato)}. "
                   f"Taxa contínua equivalente na eq. (4.5): ln(1+i_f) = {pct(math.log1p(fiat_exato))} a.a.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=serie_p, name="⟨P⟩ nominal (base 100)", line=dict(color=COR["G"])))
        fig.add_trace(go.Scatter(x=t, y=serie_s, name="σ_T produção (base 100)", line=dict(color=COR["F"])))
        fig.add_trace(go.Scatter(x=t, y=100 * serie_v, name="v × 100", line=dict(color=COR["pi"], dash="dash")))
        fig.update_layout(title="Moeda, produção e a razão v", xaxis_title="ano",
                          yaxis_title="índice", yaxis_type="log" if escala_log else "linear")
        mostrar(fig, 380)

    st.markdown("**Onde a linearização quebra.** Mesma produção, crescimentos monetários cada vez maiores:")
    amostras = np.array([0.03, 0.10, 0.30, 1.0, 5.0, 50.0])
    tabela(pd.DataFrame({
        "crescimento de ⟨P⟩": [pct(m, 0) for m in amostras],
        "i_f exato": [pct((1 + m) / (1 + g_det) - 1) for m in amostras],
        "i_f linear (eq. 4.8)": [pct(m - g_det) for m in amostras],
        "erro": [pct((m - g_det) - ((1 + m) / (1 + g_det) - 1)) for m in amostras],
    }))
    st.info(
        "A eq. (4.8) é uma expansão para variações pequenas. Em economias estáveis o erro é de "
        "centésimos de ponto percentual; em hiperinflação com produção em queda ele passa de dezenas "
        "de pontos. Por isso este app sempre calcula os índices pela forma exata."
    )

# ============================================================================
# ABA 4 - MONTE CARLO
# ============================================================================
with aba_mc:
    st.markdown("### Milhares de economias sintéticas")
    st.caption(
        "Os índices H, F, K, G seguem as eqs. (4.13) a (4.16). O repasse para os preços "
        "(defasagem e pass-through) é uma hipótese didática deste app, inspirada na discussão da seção 5.5.5."
    )
    controles, graficos = st.columns([1, 3])
    with controles:
        st.selectbox("Cenário pronto", list(PRESETS), key="preset", on_change=aplicar_preset,
                     help="Caricaturas didáticas, não calibrações. Ajuste livremente depois de escolher.")
        with st.expander("Produção", expanded=False):
            st.slider("Crescimento médio da produção (% a.a.)", -5.0, 10.0, step=0.5, key="p_g")
            st.slider("Volatilidade da produção (p.p.)", 0.0, 10.0, step=0.5, key="p_sig_g")
        with st.expander("Moeda", expanded=True):
            st.slider("Crescimento médio de M0 (% a.a.)", -5.0, 100.0, step=0.5, key="p_mu0")
            st.slider("Degrau entre agregados (p.p.)", 0.0, 10.0, step=0.5, key="p_spread",
                      help="M1 cresce M0 + degrau, M2 cresce M0 + 2·degrau, e assim por diante.")
            st.slider("Volatilidade monetária (p.p.)", 0.0, 40.0, step=0.5, key="p_sig_m")
            rho_m = st.slider("Correlação entre agregados", 0.0, 1.0, 0.7, 0.05)
        with st.expander("Choque (tipo 2008 ou COVID)", expanded=False):
            anos_mc = st.slider("Horizonte (anos)", 15, 60, 30, 5)
            ano_choque = st.slider("Ano do choque (0 = sem choque)", 0, anos_mc, min(25, anos_mc - 4))
            st.slider("Expansão monetária extra no choque (p.p.)", 0.0, 300.0, step=1.0, key="p_choque_m")
            st.slider("Queda da produção no choque (p.p.)", 0.0, 30.0, step=1.0, key="p_choque_y")
        with st.expander("Repasse aos preços (hipótese do app)", expanded=True):
            motor = st.selectbox("Índice que pressiona os preços", INDICES, index=2)
            st.slider("Intensidade do repasse, θ", 0.0, 2.0, step=0.05, key="p_theta",
                      help="θ = 1: toda a pressão monetária vira inflação. θ ≈ 0: transmissão bloqueada.")
            st.slider("Atraso médio, τ (anos)", 0.0, 3.0, step=0.5, key="p_tau",
                      help="Pesos de Poisson com média τ sobre as defasagens 0 a 5 anos.")
            st.slider("Ruído próprio da inflação (p.p.)", 0.0, 20.0, step=0.1, key="p_sig_pi",
                      help="Choques de oferta, câmbio, energia: tudo que não vem do desequilíbrio monetário.")
        with st.expander("Monte Carlo", expanded=False):
            n_cen = st.slider("Número de cenários", 500, 10000, 2000, 500)
            semente_mc = st.number_input("Semente", 0, 9999, 2026, 1, key="semente_mc")

    ss = st.session_state
    sim = simular(n_cen, anos_mc, int(semente_mc), ss.p_g / 100, ss.p_sig_g / 100, ss.p_mu0 / 100,
                  ss.p_spread / 100, ss.p_sig_m / 100, rho_m, ano_choque, ss.p_choque_m / 100,
                  ss.p_choque_y / 100, INDICES.index(motor), ss.p_theta, ss.p_tau, ss.p_sig_pi / 100)
    ind, infl = sim["I"], sim["pi"]
    eixo_anos = np.arange(1, anos_mc + 1)

    with graficos:
        st.latex(r"\pi_t=\theta\sum_{h=0}^{5}w_h(\tau)\;\Pi^{(M)}_{t-h}+\varepsilon_t"
                 r"\qquad\text{(hipótese didática)}")
        colunas = st.columns(5)
        for coluna, nome, k in zip(colunas, INDICES, range(4)):
            coluna.metric(f"{nome} médio ({AGREGADOS[k]})", pct(ind[k].mean()))
        colunas[4].metric("Inflação média", pct(infl.mean()))

        # --- trajetória individual, no estilo das figuras 1 a 5 do artigo
        cenario = st.slider("Cenário exibido", 1, n_cen, 1) - 1
        faixa = st.checkbox("Mostrar faixa 10 a 90 % da inflação entre cenários", value=True)
        fig = go.Figure()
        if faixa:
            p10, p90 = np.percentile(infl, [10, 90], axis=0)
            fig.add_trace(go.Scatter(x=eixo_anos, y=100 * p90, line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=eixo_anos, y=100 * p10, fill="tonexty", line=dict(width=0),
                                     fillcolor="rgba(148,103,189,0.18)", name="inflação 10 a 90 %",
                                     hoverinfo="skip"))
        for k, nome in enumerate(INDICES):
            fig.add_trace(go.Scatter(x=eixo_anos, y=100 * ind[k, cenario], name=f"{nome} ({AGREGADOS[k]})",
                                     line=dict(color=COR[nome], width=1.8)))
        fig.add_trace(go.Scatter(x=eixo_anos, y=100 * infl[cenario], name="inflação π",
                                 line=dict(color=COR["pi"], width=3)))
        if ano_choque > 0:
            fig.add_vline(x=ano_choque, line_dash="dash", line_color="gray", annotation_text="choque")
        fig.update_layout(title=f"Cenário {cenario + 1}: índices e inflação", xaxis_title="ano",
                          yaxis_title="% a.a.")
        mostrar(fig, 430)

        # --- correlações defasadas (análogo da Tabela 1)
        esquerda, direita = st.columns(2)
        h_max = 4
        with esquerda:
            fig = go.Figure()
            melhor = {}
            for k, nome in enumerate(INDICES):
                rho = correlacao_defasada(ind[k], infl, h_max)
                media_rho = np.nanmean(rho, axis=1)
                melhor[nome] = (int(np.nanargmax(media_rho)), rho)
                fig.add_trace(go.Bar(x=[f"h = {h}" for h in range(h_max + 1)], y=media_rho,
                                     name=nome, marker_color=COR[nome]))
            fig.update_layout(title="Corr(índice em t, inflação em t+h), média dos cenários",
                              yaxis_title="ρ", barmode="group", hovermode="closest")
            mostrar(fig, 380)
            h_otimo, rho_motor = melhor[motor]
            q10, q90 = np.nanpercentile(rho_motor[h_otimo], [10, 90])
            st.caption(
                f"Para {motor}, o pico médio está em h = {h_otimo}. Entre cenários, 80 % das correlações "
                f"nesse h ficam entre {num(q10, 2)} e {num(q90, 2)}: essa é a incerteza amostral de uma "
                f"série com {anos_mc} pontos anuais. Para comparação, o artigo reporta para a Alemanha cerca de 0,3 em "
                "h = 0, 0,67 em h = 1 e 0,76 em h = 2."
            )

        # --- distribuição entre cenários e momentos estatísticos
        with direita:
            estatisticas = {
                "Inflação média no período": 100 * infl.mean(axis=1),
                "Pico de inflação no período": 100 * infl.max(axis=1),
                f"v final / v inicial ({AGREGADOS[INDICES.index(motor)]})":
                    np.prod(1 + ind[INDICES.index(motor)], axis=1),
            }
            escolha = st.selectbox("Grandeza resumida por cenário", list(estatisticas))
            amostra = estatisticas[escolha]
            fig = go.Figure(go.Histogram(x=amostra, nbinsx=60, histnorm="probability density",
                                         marker_color=COR["pi"], opacity=0.75))
            fig.update_layout(title="Distribuição entre cenários", xaxis_title=escolha,
                              yaxis_title="densidade", hovermode="closest")
            mostrar(fig, 330)
            media_a, var_a, assim_a, curt_a = momentos_estatisticos(amostra)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Média", num(media_a, 2))
            e2.metric("Variância", num(var_a, 2))
            e3.metric("Assimetria", num(assim_a, 2))
            e4.metric("Curtose exc.", num(curt_a, 2))
            st.caption("Momentos **estatísticos** da distribuição entre cenários. Não são os "
                       "“momentos de inflação” H, F, K, G do artigo.")

        # --- regimes de crescimento
        regimes = classificar_regime(sim["gy"], ind[INDICES.index(motor)])
        nomes_regime, contagem = np.unique(regimes, return_counts=True)
        ordem = np.argsort(contagem)
        fig = go.Figure(go.Bar(x=100 * contagem[ordem] / contagem.sum(), y=nomes_regime[ordem],
                               orientation="h", marker_color=COR["F"]))
        fig.update_layout(title=f"Frequência dos regimes de crescimento (seção 6.2), usando {motor}",
                          xaxis_title="% dos anos simulados", hovermode="closest")
        mostrar(fig, 300)

    with st.expander("🧪 Roteiro de exploração"):
        st.markdown(
            "1. **Atraso.** No cenário estável, varie τ de 0 a 3 e observe o pico das barras de correlação "
            "andar de h = 0 para h = 2 ou 3. É assim que o artigo infere o atraso de um a dois anos.\n"
            "2. **Um evento só.** Coloque o ano do choque em 0. As correlações despencam? Quanto da "
            "evidência depende de um único episódio grande?\n"
            "3. **Japão e China.** Escolha “Transmissão bloqueada”. Os índices continuam positivos e a "
            "inflação fica perto de zero. O modelo distingue isso de θ pequeno? O que seria θ na economia real?\n"
            "4. **Hiperinflação.** Escolha “Hiperinflação” e compare o histograma da *média* com o do "
            "*pico*. Qual tem cauda mais pesada, e por que a média esconde as caudas?\n"
            "5. **Hierarquia.** Zere o degrau entre agregados. A ordenação G > K > F > H some. O que a "
            "ordenação persistente relatada no artigo exige das razões M3/M2, M2/M1 e M1/M0?"
        )
    with st.expander("Respostas comentadas"):
        st.markdown(
            "2. Sem o choque, índice e inflação só compartilham ruído de pequena amplitude e a correlação "
            "cai bastante. O próprio artigo relata que, restringindo a amostra alemã a 1995–2019, os "
            "índices defasados perdem significância.\n\n"
            "4. A média temporal soma muitos anos e, pelo teorema central do limite, tende à gaussiana "
            "(curtose perto de zero). O pico é uma estatística de extremos e preserva a cauda.\n\n"
            "5. G > K sempre significa M3/M2 crescendo todos os anos; idem para os outros pares. É uma "
            "regularidade forte sobre a composição da moeda, que vale conferir diretamente nas séries brutas."
        )

# ============================================================================
# ABA 5 - DADOS REAIS
# ============================================================================
with aba_dados:
    st.markdown("### Aplique as fórmulas aos seus dados")
    st.markdown(
        "Envie um CSV anual com as colunas **ano**, **PIB** e pelo menos um entre **M0, M1, M2, M3**. "
        "A coluna **inflacao** (em % a.a.) é opcional, mas necessária para as correlações. "
        "Separador vírgula ou ponto e vírgula; decimal com ponto ou vírgula. "
        "Fontes usuais: Banco Central do Brasil (SGS), IBGE, FRED, Banco Mundial."
    )
    exemplo = exemplo_sintetico()
    esquerda, direita = st.columns([2, 1])
    with esquerda:
        arquivo = st.file_uploader("Arquivo CSV", type=["csv", "txt"])
    with direita:
        st.download_button("Baixar modelo de CSV (dados sintéticos)",
                           exemplo.to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
                           file_name="modelo_indices.csv", mime="text/csv")
        usar_exemplo = st.checkbox("Usar o exemplo sintético", value=arquivo is None)

    dados, avisos = None, []
    if arquivo is not None:
        dados, avisos = ler_csv(arquivo.getvalue())
    elif usar_exemplo:
        dados = exemplo
        st.info("Exibindo **dados sintéticos** gerados pelo simulador. Não representam nenhum país.")
    for aviso in avisos:
        st.warning(aviso)

    if dados is not None:
        resultado = calcular_indices(dados)
        presentes = [n for n in INDICES if n in resultado]
        tem_inflacao = "inflacao" in resultado and resultado["inflacao"].notna().sum() >= 4
        st.caption("Convenção: o índice do ano t mede a variação de M/PIB entre t−1 e t.")

        eixo_duplo = st.checkbox("Inflação em eixo secundário (útil em inflação alta)", value=False)
        fig = go.Figure()
        for nome in presentes:
            fig.add_trace(go.Scatter(x=resultado["ano"], y=resultado[nome], name=nome,
                                     line=dict(color=COR[nome])))
        if tem_inflacao:
            fig.add_trace(go.Scatter(x=resultado["ano"], y=resultado["inflacao"], name="inflação",
                                     line=dict(color=COR["pi"], width=3),
                                     yaxis="y2" if eixo_duplo else "y"))
        fig.update_layout(title="Índices monetário-produtivos e inflação observada",
                          xaxis_title="ano", yaxis_title="% a.a.",
                          yaxis2=dict(title="inflação (% a.a.)", overlaying="y", side="right"))
        mostrar(fig, 420)

        if tem_inflacao:
            esquerda, direita = st.columns(2)
            with esquerda:
                st.markdown("**Correlações defasadas, Corr(I_t, π_{t+h})** (eq. 5.1)")
                linhas = []
                for nome in presentes:
                    linha = {"índice": nome}
                    for h in range(4):
                        par = pd.concat([resultado[nome], resultado["inflacao"].shift(-h)], axis=1).dropna()
                        linha[f"h = {h}"] = num(par.iloc[:, 0].corr(par.iloc[:, 1]), 2) if len(par) >= 6 else "n/d"
                    linhas.append(linha)
                tabela(pd.DataFrame(linhas))
            with direita:
                st.markdown("**Regressão preditiva** (eqs. 5.4 e 5.5): π_t = α + ρ π_{t−1} + β I_{t−1}")
                linhas = []
                for nome in presentes:
                    base = pd.DataFrame({"y": resultado["inflacao"],
                                         "y1": resultado["inflacao"].shift(1),
                                         "i1": resultado[nome].shift(1)}).dropna()
                    if len(base) < 8:
                        continue
                    _, r2_ar = mqo(base[["y1"]].to_numpy(), base["y"].to_numpy())
                    beta, r2_ind = mqo(base[["y1", "i1"]].to_numpy(), base["y"].to_numpy())
                    linhas.append({"índice": nome, "β": num(beta[2], 3), "R² só AR(1)": num(r2_ar, 3),
                                   "R² com índice": num(r2_ind, 3), "n": len(base)})
                if linhas:
                    tabela(pd.DataFrame(linhas))
                else:
                    st.caption("São necessários pelo menos 8 anos com inflação para a regressão.")
            st.warning(
                "Com poucas dezenas de pontos anuais, um único episódio (como 2020–2022) pode dominar "
                "correlações e R². R² dentro da amostra sempre sobe ao acrescentar um regressor. "
                "Trate estes números como exploração, não como teste."
            )
        with st.expander("Ver tabela de índices calculados"):
            tabela(resultado.round(3))
            st.download_button("Baixar índices (CSV)",
                               resultado.to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
                               file_name="indices_HFKG.csv", mime="text/csv")

# ============================================================================
# ABA 6 - DEBATE
# ============================================================================
with aba_debate:
    st.markdown("### Questões em aberto e extensões possíveis")
    st.markdown(
        "O que os laboratórios deste app sugerem como próximos passos. São leituras de quem explorou "
        "o modelo numericamente, oferecidas como contribuição à discussão."
    )
    st.markdown(
        "**1. Heterogeneidade setorial como fonte de v.** A eq. (3.5) vale em primeira ordem em c. "
        "Indo à segunda ordem aparece o termo c²⟨(D−S)²⟩/2 (aba 2, experimento 2). Isso abre uma "
        "previsão própria do modelo: economias com desequilíbrios setoriais mais dispersos teriam v "
        "maior, mesmo com oferta e demanda equilibradas na média. Dá para testar com dados setoriais?\n\n"
        "**2. Ponte com a teoria quantitativa.** Pela equação de trocas, MV = PY. Com σ_T igual ao PIB "
        "real, v = P/V e a inflação fiat é π − ΔV/V; com o PIB nominal, v = 1/V. Explicitar essa ponte, "
        "e qual PIB entra em σ_T, ajudaria o leitor de economia a situar o arcabouço. A velocidade de "
        "circulação seria uma candidata natural à variável oculta do atraso discutido na seção 5?\n\n"
        "**3. Um observável para a parte termodinâmica.** Na aplicação empírica, σ_T é aproximado pelo "
        "PIB, e os termos b_k, m_k, b_A e |ΔS| cancelam em v (aba 2, experimento 3). Uma extensão "
        "natural é medir σ_T com dados físicos, como exergia ou fluxo de materiais, e comparar o v "
        "resultante com o monetário. Seria o teste direto da contribuição original do arcabouço.\n\n"
        "**4. Construção das séries.** A hierarquia G > K > F > H preservada por 30 anos implica "
        "M3/M2, M2/M1 e M1/M0 crescendo ano após ano, um fato estilizado interessante por si só. "
        "Um apêndice de dados facilitaria a reprodução: por exemplo, como entram o fim da publicação "
        "do M3 pelo Fed em 2006 e a redefinição do M1 norte-americano em 2020.\n\n"
        "**5. Robustez entre regimes.** A seção 5.5.5 já aponta que o conteúdo preditivo se concentra "
        "em episódios de grande expansão monetária e é fraco em 1995–2019 (aba 4, experimento 2). "
        "Dados trimestrais, um painel de países ou episódios anteriores, como os anos 1970, "
        "ampliariam a amostra e permitiriam separar o efeito de 2020–2022.\n\n"
        "**6. Estoque de ativos.** A aproximação f(t) ≈ 0 é assumida como primeiro passo, e o artigo "
        "reconhece que os ativos antigos importam. Religar f(t) poderia dar conta de Japão e China: "
        "moeda nova absorvida por ativos já existentes, em vez de pressionar os preços ao consumidor.\n\n"
        "**7. Nomenclatura.** Como “momentos” também nomeia média, variância, assimetria e curtose, uma "
        "nota no texto evitaria confusão. Existe uma estrutura de expansão, como em multipolos, por "
        "trás do nome?"
    )
    st.divider()
    st.caption(
        "Referência: Marinho, A. & Rocha, H. (2026). Analyses of Inflation Dynamics as a Consequence of "
        f"Monetary-Production Imbalance in Global Markets. Computational Economics. {DOI_URL}"
    )
