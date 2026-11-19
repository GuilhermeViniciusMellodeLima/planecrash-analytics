"""
Página: Comparação dos Modelos de Machine Learning - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st
import pandas as pd

from src.ui_components import inject_custom_css, render_header, render_sidebar_info, render_metric_card
from src.train import are_models_trained, load_saved_artifacts
from src.visualizations import (
    plot_metrics_comparison,
    plot_individual_metric_bar,
    plot_confusion_matrix_heatmap,
    plot_combined_roc_curves,
    plot_feature_importances
)

st.set_page_config(
    page_title="PLANECRASH ANALYTICS — Comparação dos Modelos",
    page_icon="⚖️",
    layout="wide"
)

inject_custom_css()
render_sidebar_info()

render_header(
    title="Comparação de Desempenho dos Modelos",
    subtitle="Avaliação Comparativa de Acurácia, Precisão, Recall, F1-Score, ROC-AUC e Matrizes de Confusão (Entregável 2)"
)

if not are_models_trained():
    st.warning("⚠️ Os modelos ainda não foram treinados. Por favor, acerte o treinamento na página **Treinamento dos Modelos** primeiro.")
    if st.button("Ir para Treinamento dos Modelos"):
        st.switch_page("pages/3_Treinamento.py")
    st.stop()

artifacts = load_saved_artifacts()
if not artifacts:
    st.error("Erro ao carregar os artefatos salvos dos modelos.")
    st.stop()

df_comp = artifacts.get("comparison_table", pd.DataFrame())
results = artifacts.get("results", {})
diag = artifacts.get("diagnosis", {})

# Tabela Oficial Comparativa do Entregável 2
st.markdown("### 📊 Tabela Comparativa Oficial dos Modelos")
st.markdown(
    "Métricas calculadas exclusivamente sobre as **5.515 amostras inéditas do conjunto de teste (20%)**, com a Classe 1 (Acidentes Fatais) tratada como positiva:"
)

st.dataframe(
    df_comp.style.format({
        "Acurácia": "{:.2%}",
        "Precisão": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}",
        "ROC-AUC": "{:.2%}"
    }).highlight_max(subset=["Recall", "F1-Score", "ROC-AUC"], color="#1E3A8A"),
    use_container_width=True,
    hide_index=True
)

# Conclusão e Veredito Técnico Automático
st.markdown("""
<div style="background: linear-gradient(135deg, #131E33 0%, #172554 100%); border: 1px solid #1D4ED8; border-radius: 12px; padding: 1.4rem 1.8rem; margin: 1.5rem 0;">
    <div style="font-size: 0.8rem; text-transform: uppercase; color: #60A5FA; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 6px;">
        🏆 Veredito Técnico e Diagnóstico do Algoritmo Vencedor
    </div>
    <h3 style="color: #F8FAFC; margin: 0 0 10px 0;">
        Melhor Desempenho Global: <span style="color: #00D2FF;">""" + str(diag.get("melhor_modelo", "Random Forest")) + """</span>
    </h3>
    <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6; margin: 0;">
        """ + str(diag.get("explicacao_tecnica", "")) + """
    </p>
</div>
""", unsafe_allow_html=True)

# Visualizações Comparativas
tab_all, tab_single, tab_cm, tab_roc, tab_imp = st.tabs([
    "📊 Gráfico Comparativo Geral",
    "🎯 Métricas Individuais",
    "🔲 Matrizes de Confusão",
    "📈 Curvas ROC",
    "⭐ Importância das Variáveis"
])

with tab_all:
    st.plotly_chart(plot_metrics_comparison(df_comp), use_container_width=True)

with tab_single:
    st.markdown("#### Selecione uma Métrica Específica para Comparação:")
    metric_choice = st.selectbox(
        "Métrica a destacar:",
        options=["Recall", "F1-Score", "Acurácia", "Precisão", "ROC-AUC"],
        index=0,
        help="O Recall é prioritário no domínio aeronáutico pois minimiza o risco de falsos negativos."
    )
    st.plotly_chart(plot_individual_metric_bar(df_comp, metric_choice), use_container_width=True)

    if metric_choice == "Recall":
        st.info(
            "📌 **Por que o Recall é a métrica mais crítica?** Em análise de acidentes e segurança de transporte, "
            "um **Falso Negativo** significa classificar erroneamente um acidente que teve vítimas fatais como 'sem vítimas'. "
            "Esse é o pior erro operacional possível, justificando priorizar algoritmos que maximizem a sensibilidade da detecção."
        )

with tab_cm:
    st.markdown("#### Matrizes de Confusão Individuais (Conjunto de Teste - 20%)")
    cm_cols = st.columns(3)

    for idx, (name, res) in enumerate(results.items()):
        with cm_cols[idx]:
            cm_dict = res.get("metrics", {}).get("matriz_confusao", {})
            st.plotly_chart(plot_confusion_matrix_heatmap(cm_dict, name), use_container_width=True)

    st.markdown("""
    **Legenda dos Quadrantes:**
    - **Verdadeiro Negativo (VN):** Acidentes sem vítimas fatais classificados corretamente como sem vítimas.
    - **Falso Positivo (FP):** Acidentes sem vítimas fatais previstos erroneamente como fatais (alarme falso tolerável).
    - **Falso Negativo (FN):** Acidentes fatais previstos erroneamente como sem vítimas (**erro mais severo a ser minimizado**).
    - **Verdadeiro Positivo (VP):** Acidentes fatais identificados corretamente pelo algoritmo.
    """)

with tab_roc:
    st.markdown("#### Análise Discriminativa via Curva ROC (Receiver Operating Characteristic)")
    st.plotly_chart(plot_combined_roc_curves(results), use_container_width=True)
    st.caption("A área sob a curva ROC (ROC-AUC) quantifica a probabilidade de o classificador ordenar uma ocorrência fatal aleatória acima de uma ocorrência não fatal aleatória.")

with tab_imp:
    st.markdown("#### Relevância das Variáveis na Decisão dos Modelos")
    model_imp_choice = st.selectbox(
        "Selecione o modelo para inspecionar as características mais determinantes:",
        options=["Random Forest", "Árvore de Decisão", "Regressão Logística"]
    )
    fig_imp = plot_feature_importances(results, model_name=model_imp_choice)
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Pesos não disponíveis para este modelo.")
