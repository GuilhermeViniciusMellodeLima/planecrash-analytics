"""
Página: Treinamento dos Modelos de Machine Learning - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st
import pandas as pd
import time

from src.ui_components import inject_custom_css, render_header, render_sidebar_info, render_metric_card
from src.data_loader import load_raw_data
from src.train import train_all_models, are_models_trained, load_saved_artifacts
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, LEAKAGE_COLUMNS

st.set_page_config(
    page_title="PLANECRASH ANALYTICS — Treinamento",
    page_icon="⚙️",
    layout="wide"
)

inject_custom_css()
render_sidebar_info()

render_header(
    title="Treinamento dos Modelos de Machine Learning",
    subtitle="Ajuste dos Pipelines Supervisionados Obrigatórios do Entregável 2 com Validação Estratificada 80/20"
)

# Carregamento do dataset base
@st.cache_data(show_spinner=False)
def get_training_data():
    df, _ = load_raw_data(source_preference="auto")
    return df

df_raw = get_training_data()

if df_raw is None:
    st.error("Não foi possível carregar a base para treinamento.")
    st.stop()

# Explicação Metodológica
st.markdown("""
<div style="background: #131E33; border: 1px solid #233554; border-radius: 12px; padding: 1.2rem 1.6rem; margin-bottom: 1.5rem;">
    <h4 style="color: #00D2FF; margin-top: 0; margin-bottom: 8px;">Metodologia de Treinamento e Avaliação (Entregável 2)</h4>
    <p style="color: #CBD5E1; font-size: 0.92rem; margin: 0; line-height: 1.6;">
        Os três algoritmos supervisionados são treinados de forma rigorosa sob as mesmas condições amostrais:
        divisão estratificada de <strong>80% para treino</strong> e <strong>20% para teste</strong>, preservando a
        proporção natural das classes binárias. Os pipelines de imputação e codificação são calibrados <em>exclusivamente</em>
        no subconjunto de treinamento, prevenindo qualquer forma de contaminação estatística (<strong>Data Leakage</strong>).
    </p>
</div>
""", unsafe_allow_html=True)

# Painel de Configuração e Treinamento
col_conf, col_status = st.columns([1, 1])

with col_conf:
    st.markdown("#### ⚙️ Parâmetros de Execução")
    use_weight = st.checkbox(
        "Utilizar balanceamento automático de pesos (`class_weight='balanced'`)",
        value=True,
        help="Compensa a proporção entre acidentes fatais e não fatais, penalizando erros nas classes minoritárias para elevar o Recall."
    )

    st.markdown("##### Variáveis Selecionadas para os Modelos:")
    st.markdown(f"- **Numéricas ({len(NUMERIC_FEATURES)}):** `{', '.join(NUMERIC_FEATURES)}`")
    st.markdown(f"- **Categóricas ({len(CATEGORICAL_FEATURES)}):** `{', '.join(CATEGORICAL_FEATURES)}`")

    st.markdown("##### Prevenção de Data Leakage Ativa:")
    st.caption(f"Bloqueadas do vetor preditor: {', '.join(LEAKAGE_COLUMNS)}.")

    train_btn = st.button("🚀 Iniciar Treinamento dos 3 Modelos", type="primary", use_container_width=True)

with col_status:
    st.markdown("#### 📦 Status dos Modelos em Disco")
    is_trained = are_models_trained()

    if is_trained:
        st.success("✅ Modelos treinados e persistidos em formato Joblib no diretório `models/`.")
        loaded = load_saved_artifacts()
        if loaded:
            meta_info = loaded.get("metadata", {})
            st.markdown(f"""
            - **Amostras de Treino:** {meta_info.get('train_samples', 0):,} registros (80%)
            - **Amostras de Teste:** {meta_info.get('test_samples', 0):,} registros (20%)
            - **Pesos balanceados:** {'Sim' if meta_info.get('use_class_weight') else 'Não'}
            - **Algoritmos salvos:** Regressão Logística, Árvore de Decisão e Random Forest.
            """)
    else:
        st.warning("⚠️ Nenhum modelo treinado detectado em disco.")
        st.info("Clique no botão acima para iniciar o ajuste dos algoritmos.")

# Execução do Treinamento ao Clicar
if train_btn:
    with st.status("🔄 Executando pipeline completo de Machine Learning...", expanded=True) as status:
        st.write("1. Filtrando valores nulos no target e saneando anomalias históricas...")
        time.sleep(0.3)
        st.write("2. Realizando separação estratificada 80/20 (random_state=42, stratify=y)...")
        time.sleep(0.3)
        st.write("3. Treinando Regressão Logística (Baseline)...")
        time.sleep(0.4)
        st.write("4. Treinando Árvore de Decisão (DecisionTreeClassifier)...")
        time.sleep(0.4)
        st.write("5. Treinando Random Forest Classifier (Ensemble com 100 estimadores)...")
        time.sleep(0.4)
        st.write("6. Calculando predições e métricas reais no conjunto de teste de 20%...")

        payload = train_all_models(df_raw, use_class_weight=use_weight, save_on_disk=True)

        st.write("7. Serializando artefatos com Joblib e salvando relatório em JSON...")
        time.sleep(0.2)
        status.update(label="🎉 Treinamento e avaliação concluídos com sucesso!", state="complete", expanded=False)

    st.success("Treinamento finalizado! Os resultados calculados com o conjunto de teste real estão apresentados abaixo.")
    st.rerun()

# Exibição dos Resultados dos Modelos
if are_models_trained():
    artifacts = load_saved_artifacts()
    if artifacts:
        df_comp = artifacts.get("comparison_table", pd.DataFrame())
        results = artifacts.get("results", {})

        st.markdown("---")
        st.markdown("### 📋 Resultados Obtidos no Conjunto de Teste (Dados Não Vistos — 20%)")

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

        st.markdown("#### Diagnóstico de Generalização (Treino vs Teste)")
        st.caption("Verificação da presença de sobreajuste (overfitting) entre o aprendizado e o conjunto de teste:")

        rows_overfit = []
        for name, data in results.items():
            test_m = data.get("metrics", {})
            train_m = data.get("train_metrics", {})
            rows_overfit.append({
                "Modelo": name,
                "Acurácia Treino": f"{train_m.get('acuracia', 0):.2%}",
                "Acurácia Teste": f"{test_m.get('acuracia', 0):.2%}",
                "Recall Treino": f"{train_m.get('recall', 0):.2%}",
                "Recall Teste": f"{test_m.get('recall', 0):.2%}",
                "Gap de Generalização": f"{(train_m.get('acuracia', 0) - test_m.get('acuracia', 0)):.2%}"
            })

        st.dataframe(pd.DataFrame(rows_overfit), use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background: rgba(0, 210, 255, 0.08); border: 1px solid rgba(0, 210, 255, 0.25); border-radius: 8px; padding: 0.9rem 1.2rem; margin-top: 1rem;">
            <strong>Próximo passo:</strong> Acesse a aba <strong>Comparação dos Modelos</strong> no menu lateral para visualizar
            os gráficos comparativos, matrizes de confusão detalhadas e curvas ROC.
        </div>
        """, unsafe_allow_html=True)
