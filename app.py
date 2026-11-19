"""
PLANECRASH ANALYTICS — Dashboard Principal
Sistema Inteligente de Análise e Classificação de Acidentes Aeronáuticos

Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st
import pandas as pd
import os

from src.ui_components import inject_custom_css, render_header, render_sidebar_info, render_metric_card
from src.data_loader import load_raw_data, get_dataset_metadata
from src.train import are_models_trained, load_saved_artifacts
from src.visualizations import plot_target_distribution, plot_metrics_comparison

# Configuração da página Streamlit
st.set_page_config(
    page_title="PLANECRASH ANALYTICS — Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar_info()

# Cabeçalho Principal
render_header(
    title="PLANECRASH ANALYTICS",
    subtitle="Sistema Inteligente de Classificação da Gravidade de Acidentes Aeronáuticos"
)

# Carregamento em cache dos dados para a sessão
@st.cache_data(show_spinner=False)
def get_cached_data():
    df, info = load_raw_data(source_preference="auto")
    return df, info

df, load_info = get_cached_data()

# Banner descritivo e autoria acadêmica
st.markdown("""
<div style="background: #131E33; border: 1px solid #1E293B; border-radius: 12px; padding: 1.2rem 1.6rem; margin-bottom: 1.8rem;">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
        <div style="flex: 1; min-width: 300px;">
            <h4 style="color: #F8FAFC; margin: 0 0 8px 0; font-size: 1.15rem;">Sobre o Projeto Acadêmico</h4>
            <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                O <strong>PLANECRASH ANALYTICS</strong> é uma plataforma acadêmica desenvolvida para analisar registros históricos
                mundiais de acidentes aeronáuticos e classificar a gravidade de incidentes em duas categorias:
                <strong>Sem vítimas fatais (Classe 0)</strong> e <strong>Com vítimas fatais (Classe 1)</strong>, utilizando
                algoritmos supervisionados de Machine Learning e apoio de Large Language Models (LLM).
            </p>
        </div>
        <div style="background: #0A0F1D; border: 1px solid #233554; border-radius: 8px; padding: 0.8rem 1.2rem; min-width: 260px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 700; margin-bottom: 4px;">Integrantes</div>
            <div style="color: #F1F5F9; font-size: 0.88rem; font-weight: 600;">Guilherme Vinicius Mello de Lima</div>
            <div style="color: #00D2FF; font-family: monospace; font-size: 0.8rem; margin-bottom: 6px;">Prontuário: CG3028461</div>
            <div style="color: #F1F5F9; font-size: 0.88rem; font-weight: 600;">Luiz Gustavo Alves dos Santos</div>
            <div style="color: #00D2FF; font-family: monospace; font-size: 0.8rem;">Prontuário: CG302900X</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Indicadores Gerais do Dataset
st.markdown("### 📊 Indicadores Gerais da Base Histórica")

if df is not None:
    meta = get_dataset_metadata(df)
    t_summary = meta.get("target_summary", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            label="Total de Acidentes",
            value=f"{meta.get('total_registros', 0):,}".replace(",", "."),
            hint="Registros históricos catalogados"
        )
    with col2:
        render_metric_card(
            label="Variáveis / Colunas",
            value=str(meta.get("total_colunas", 0)),
            hint="Atributos técnicos e contextuais"
        )
    with col3:
        render_metric_card(
            label="Acidentes com Fatais (Classe 1)",
            value=f"{t_summary.get('classe_1_fatais', 0):,}".replace(",", "."),
            hint=f"{t_summary.get('pct_fatais', 0):.1f}% do total registrado",
            border_color="rgba(244, 63, 94, 0.4)"
        )
    with col4:
        render_metric_card(
            label="Acidentes sem Fatais (Classe 0)",
            value=f"{t_summary.get('classe_0_nao_fatais', 0):,}".replace(",", "."),
            hint=f"{100 - t_summary.get('pct_fatais', 0):.1f}% do total registrado",
            border_color="rgba(16, 185, 129, 0.4)"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráficos da Visão Geral
    row_chart1, row_chart2 = st.columns([1, 1])

    with row_chart1:
        st.plotly_chart(plot_target_distribution(df), use_container_width=True)

    with row_chart2:
        # Status e Diagnóstico dos Modelos
        st.markdown("#### 🤖 Status dos Modelos de Machine Learning")
        if are_models_trained():
            artifacts = load_saved_artifacts()
            if artifacts:
                df_comp = artifacts.get("comparison_table", pd.DataFrame())
                diag = artifacts.get("diagnosis", {})

                st.success(f"✅ Três modelos treinados e validados no conjunto de teste (20%).")
                st.dataframe(
                    df_comp.style.format({
                        "Acurácia": "{:.2%}",
                        "Precisão": "{:.2%}",
                        "Recall": "{:.2%}",
                        "F1-Score": "{:.2%}",
                        "ROC-AUC": "{:.2%}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                st.info(f"🏆 **Destaque:** {diag.get('explicacao_tecnica', '')}")
            else:
                st.warning("Modelos salvos detectados, mas não foi possível carregar métricas.")
        else:
            st.warning("⚠️ Modelos ainda não treinados nesta sessão.")
            st.markdown("Acesse a página **Treinamento dos Modelos** no menu lateral para iniciar o processamento com divisão 80/20.")

    # Atalhos para as Funcionalidades
    st.markdown("---")
    st.markdown("### 🚀 Acesso Rápido aos Módulos do Sistema")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">📁</div>
            <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Dataset Oficial</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">Inspecione dicionário, valores ausentes e integridade da base.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ir para Dataset", key="btn_ds"):
            st.switch_page("pages/1_Dataset.py")

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">📈</div>
            <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Análise Exploratória</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">Explore gráficos dinâmicos por fase de voo, tipo e região.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ir para Análise Exploratória", key="btn_eda"):
            st.switch_page("pages/2_Analise_Exploratoria.py")

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">🎯</div>
            <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Simular Predição</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">Preveja a gravidade de um cenário com os modelos treinados.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ir para Predição", key="btn_pred"):
            st.switch_page("pages/5_Predicao.py")

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">💡</div>
            <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Análise com IA (LLM)</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">Interprete resultados e gere resumos acadêmicos com IA.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ir para Análise com IA", key="btn_ai"):
            st.switch_page("pages/6_Analise_IA.py")

else:
    st.error("Não foi possível carregar a base de dados. Verifique a conexão com a internet ou adicione o arquivo local.")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("PLANECRASH ANALYTICS © 2026 — Projeto Acadêmico desenvolvido por Guilherme Vinicius Mello de Lima e Luiz Gustavo Alves dos Santos.")
