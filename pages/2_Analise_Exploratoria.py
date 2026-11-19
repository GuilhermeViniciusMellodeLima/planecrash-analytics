"""
Página: Análise Exploratória dos Dados (EDA) - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.ui_components import inject_custom_css, render_header, render_sidebar_info, render_metric_card
from src.data_loader import load_raw_data
from src.visualizations import (
    plot_target_distribution,
    plot_flight_phase_distribution,
    plot_flight_type_distribution,
    plot_geographic_distribution,
    plot_people_on_board_vs_fatalities,
    plot_accidents_over_time
)

st.set_page_config(
    page_title="PLANECRASH ANALYTICS — Análise Exploratória",
    page_icon="📈",
    layout="wide"
)

inject_custom_css()
render_sidebar_info()

render_header(
    title="Análise Exploratória de Dados (EDA)",
    subtitle="Padrões Históricos da Gravidade de Acidentes Aeronáuticos e Fatores de Risco Operacional"
)

@st.cache_data(show_spinner=False)
def get_data():
    df, _ = load_raw_data(source_preference="auto")
    return df

df_raw = get_data()

if df_raw is None:
    st.error("Erro ao carregar dados para análise exploratória.")
    st.stop()

# Filtros Interativos na Barra Lateral ou Expander Superior
with st.expander("🔍 Filtros Dinâmicos de Análise", expanded=False):
    f_col1, f_col2, f_col3 = st.columns(3)

    regioes_disponiveis = ["Todas"] + sorted([str(r) for r in df_raw["region"].dropna().unique()])
    with f_col1:
        sel_region = st.selectbox("Filtrar por Região:", regioes_disponiveis, index=0)

    fases_disponiveis = ["Todas"] + sorted([str(p) for p in df_raw["flight_phase"].dropna().unique() if p])
    with f_col2:
        sel_phase = st.selectbox("Filtrar por Fase de Voo:", fases_disponiveis, index=0)

    tipos_disponiveis = ["Todos"] + sorted([str(t) for t in df_raw["flight_type"].dropna().unique() if t])
    with f_col3:
        sel_type = st.selectbox("Filtrar por Tipo de Voo:", tipos_disponiveis, index=0)

# Aplicação dos Filtros
df_filtered = df_raw.copy()
if sel_region != "Todas":
    df_filtered = df_filtered[df_filtered["region"] == sel_region]
if sel_phase != "Todas":
    df_filtered = df_filtered[df_filtered["flight_phase"] == sel_phase]
if sel_type != "Todos":
    df_filtered = df_filtered[df_filtered["flight_type"] == sel_type]

# Resumo Estatístico do Filtro
valid_target = df_filtered["total_fatalities"].dropna()
total_filtered = len(valid_target)
fatais_filtered = int((valid_target > 0).sum())
pct_fatais_filtered = (fatais_filtered / total_filtered * 100) if total_filtered > 0 else 0

st.markdown("#### Indicadores do Recorte Selecionado")
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Acidentes no Recorte", f"{total_filtered:,}".replace(",", "."))
with c2:
    render_metric_card("Ocorrências Fatais", f"{fatais_filtered:,}".replace(",", "."), border_color="rgba(244, 63, 94, 0.4)")
with c3:
    render_metric_card("Ocorrências Não Fatais", f"{total_filtered - fatais_filtered:,}".replace(",", "."), border_color="rgba(16, 185, 129, 0.4)")
with c4:
    render_metric_card("Taxa de Fatalidade", f"{pct_fatais_filtered:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Painel Gráfico 1: Target e Fases de Voo
g_col1, g_col2 = st.columns([1, 1])

with g_col1:
    st.plotly_chart(plot_target_distribution(df_filtered), use_container_width=True)

with g_col2:
    st.plotly_chart(plot_flight_phase_distribution(df_filtered), use_container_width=True)

# Painel Gráfico 2: Tipos de Operação e Regiões
g_col3, g_col4 = st.columns([1, 1])

with g_col3:
    st.plotly_chart(plot_flight_type_distribution(df_filtered, top_n=8), use_container_width=True)

with g_col4:
    st.plotly_chart(plot_geographic_distribution(df_filtered), use_container_width=True)

# Painel Gráfico 3: Pessoas a Bordo e Evolução Histórica
g_col5, g_col6 = st.columns([1, 1])

with g_col5:
    st.plotly_chart(plot_people_on_board_vs_fatalities(df_filtered), use_container_width=True)

with g_col6:
    fig_time = plot_accidents_over_time(df_filtered)
    if fig_time:
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Série temporal não disponível para o filtro atual.")

# Síntese dos Padrões Identificados
st.markdown("---")
st.markdown("### 💡 Principais Padrões Identificados na Base Histórica")

p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-weight: 700; color: #00D2FF; margin-bottom: 6px;">1. Influência Crítica da Fase de Voo</div>
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5;">
            Acidentes ocorridos durante <strong>voo em rota (Flight / En route)</strong> apresentam letalidade
            desproporcionalmente maior (>85%) devido à alta energia cinética e altitude de cruzeiro.
            Em contrapartida, incidentes em <strong>táxi, estacionamento ou aproximação final</strong>
            possuem maiores taxas de sobrevivência total.
        </div>
    </div>
    """, unsafe_allow_html=True)

with p_col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-weight: 700; color: #00D2FF; margin-bottom: 6px;">2. Localização e Infraestrutura</div>
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5;">
            Ocorrências no perímetro de <strong>aeródromos (< 10 km)</strong> contam com pronta intervenção das
            equipes de combate a incêndio e socorro especializado. Já eventos em <strong>terrenos montanhosos,
            lagos ou oceanos</strong> registram severidade muito mais elevada devido ao isolamento geográfico.
        </div>
    </div>
    """, unsafe_allow_html=True)

with p_col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-weight: 700; color: #00D2FF; margin-bottom: 6px;">3. Evolução Histórica da Aviação</div>
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5;">
            Embora o volume global de voos comerciais tenha se multiplicado exponencialmente nas últimas décadas,
            a taxa proporcional de acidentes fatais decresceu acentuadamente, impulsionada por melhorias em
            <strong>aviônica, redundância de sistemas (fly-by-wire), meteorologia por radar e treinamentos CRM</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
