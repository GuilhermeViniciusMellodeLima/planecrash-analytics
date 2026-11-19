"""
Página: Inspeção do Dataset e Dicionário de Dados - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.ui_components import inject_custom_css, render_header, render_sidebar_info, render_metric_card
from src.data_loader import load_raw_data, get_dataset_metadata
from src.preprocessing import LEAKAGE_COLUMNS, POST_CRASH_OR_META_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES

st.set_page_config(
    page_title="PLANECRASH ANALYTICS — Dataset",
    page_icon="📁",
    layout="wide"
)

inject_custom_css()
render_sidebar_info()

render_header(
    title="Inspeção da Base de Dados",
    subtitle="Carregamento, Dicionário de Variáveis, Verificação de Qualidade e Prevenção de Leakage"
)

# Painel de controle de carregamento
with st.expander("⚙️ Opções de Carregamento da Fonte de Dados", expanded=False):
    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        source_choice = st.radio(
            "Origem prioritária dos dados:",
            options=["Automático (Hugging Face com Fallback Local)", "Hugging Face Hub Direto", "Cache Local (CSV)"],
            index=0
        )
    with col_opt2:
        uploaded = st.file_uploader("Ou envie seu próprio arquivo CSV:", type=["csv"])

# Mapeia escolha da origem
source_map = {
    "Automático (Hugging Face com Fallback Local)": "auto",
    "Hugging Face Hub Direto": "huggingface",
    "Cache Local (CSV)": "local"
}

with st.spinner("Carregando base de dados..."):
    df, info = load_raw_data(
        source_preference=source_map[source_choice],
        uploaded_file=uploaded
    )

if df is not None and info.get("success", False):
    st.success(f"✅ **Status:** {info.get('message')}")
    meta = get_dataset_metadata(df)
    t_sum = meta.get("target_summary", {})

    # Indicadores
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total de Registros", f"{meta.get('total_registros', 0):,}".replace(",", "."))
    with c2:
        render_metric_card("Total de Atributos", str(meta.get("total_colunas", 0)))
    with c3:
        render_metric_card("Registros Duplicados", str(meta.get("duplicatas_completas", 0)), hint="Linhas 100% idênticas")
    with c4:
        render_metric_card("Uso de Memória", f"{meta.get('memoria_mb', 0)} MB")

    st.markdown("<br>", unsafe_allow_html=True)

    # Abas organizadas
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Amostra dos Dados",
        "📖 Dicionário de Dados & Papel no ML",
        "🔍 Valores Ausentes & Tipos",
        "⚠️ Diagnóstico de Inconsistências & Data Leakage"
    ])

    with tab1:
        st.markdown("#### Visualização dos Registros Históricos")
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            num_rows = st.slider("Quantidade de linhas a exibir:", 5, 100, 15)
        with col_ctrl2:
            cols_selected = st.multiselect(
                "Filtrar colunas para visualização:",
                options=df.columns.tolist(),
                default=["id", "aircraft", "flight_phase", "flight_type", "region", "crew_on_board", "passengers_on_board", "total_fatalities"]
            )

        if cols_selected:
            st.dataframe(df[cols_selected].head(num_rows), use_container_width=True)
        else:
            st.dataframe(df.head(num_rows), use_container_width=True)

    with tab2:
        st.markdown("#### Dicionário Oficial do Dataset PlaneCrashData e Tratamento")
        st.markdown(
            "Cada coluna possui um papel estrito no pipeline para assegurar validade científica e evitar vazamento de dados:"
        )

        dicionario_base = [
            ("id", "Identificador único do incidente", "Metadado", "Excluído da predição"),
            ("aircraft", "Modelo ou tipo da aeronave", "Feature Categórica", "Utilizada no modelo"),
            ("operator", "Companhia aérea responsável pelo voo", "Feature Textual", "Excluída (alta cardinalidade/ruído)"),
            ("registration", "Prefixo de matrícula da aeronave", "Metadado", "Excluído da predição"),
            ("flight_phase", "Fase do voo quando o incidente ocorreu", "Feature Categórica", "Utilizada no modelo"),
            ("flight_type", "Categoria operacional do voo", "Feature Categórica", "Utilizada no modelo"),
            ("survivors", "Número de sobreviventes do voo", "Data Leakage", "PROIBIDA (revela desfecho pós-queda)"),
            ("crash_site", "Terreno/local onde o incidente ocorreu", "Feature Categórica", "Utilizada no modelo"),
            ("schedule", "Programação / rota do voo", "Metadado", "Excluída da predição"),
            ("msn", "Número de série de fabricação da aeronave", "Metadado", "Excluído da predição"),
            ("yom", "Ano de fabricação do avião", "Feature Numérica", "Utilizada no modelo (com limpeza)"),
            ("crash_location", "Localização detalhada da ocorrência", "Metadado", "Excluída (ruído textual livre)"),
            ("country", "País da ocorrência", "Feature Categórica", "Utilizada em exploração"),
            ("region", "Região / continente do evento", "Feature Categórica", "Utilizada no modelo"),
            ("crew_on_board", "Número de tripulantes a bordo", "Feature Numérica", "Utilizada no modelo"),
            ("crew_fatalities", "Número de fatalidades com tripulantes", "Data Leakage", "PROIBIDA (vazamento direto)"),
            ("passengers_on_board", "Número de passageiros a bordo", "Feature Numérica", "Utilizada no modelo"),
            ("passengers_fatalities", "Fatalidades entre passageiros", "Data Leakage", "PROIBIDA (vazamento direto)"),
            ("other_fatalities", "Fatalidades em terra / não ocupantes", "Data Leakage", "PROIBIDA (vazamento direto)"),
            ("total_fatalities", "Número total de fatalidades registradas", "Target (Alvo)", "TRANSFORMADA EM CLASSE BINÁRIA (0 e 1)"),
            ("circumstances", "Narrativa descritiva detalhada do evento", "Data Leakage", "PROIBIDA (relatório pós-fato)"),
            ("crash_cause", "Causa primária apurada na investigação", "Data Leakage", "PROIBIDA (laudo pós-fato)"),
            ("timestamp", "Momento registrado em formato ISO-8601", "Metadado", "Utilizado apenas em séries temporais"),
            ("latitude", "Coordenada latitudinal decimal", "Metadado", "Excluída da modelagem direta"),
            ("longitude", "Coordenada longitudinal decimal", "Metadado", "Excluída da modelagem direta"),
            ("image_url", "URL de registro fotográfico", "Metadado", "Excluído da predição"),
            ("hex_code", "Código transponder ICAO 24-Bit Mode S", "Metadado", "Excluído da predição"),
            ("external_id", "ID de referência em bases externas", "Metadado", "Excluído da predição")
        ]

        df_dict = pd.DataFrame(dicionario_base, columns=["Coluna", "Descrição Original", "Classificação", "Tratamento no Pipeline"])

        def color_rule(val):
            if "PROIBIDA" in val:
                return "background-color: rgba(244, 63, 94, 0.2); color: #FB7185; font-weight: bold;"
            elif "TRANSFORMADA" in val:
                return "background-color: rgba(0, 210, 255, 0.2); color: #00D2FF; font-weight: bold;"
            elif "Utilizada no modelo" in val:
                return "background-color: rgba(16, 185, 129, 0.2); color: #34D399; font-weight: bold;"
            return ""

        st.dataframe(df_dict.style.applymap(color_rule, subset=["Tratamento no Pipeline"]), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Detalhamento Técnico das Colunas, Tipos e Nulos")
        col_detalhes = meta.get("colunas_detalhes", pd.DataFrame())
        st.dataframe(col_detalhes, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("#### Auditoria de Qualidade dos Dados e Prevenção de Data Leakage")

        st.markdown("""
        > [!IMPORTANT]
        > **Prevenção Rigorosa de Data Leakage:**
        > Para que um modelo de Machine Learning tenha validade prática e preditiva, ele **não pode ter acesso a variáveis
        > que revelem o desfecho do acidente**. Colunas como `survivors`, `crew_fatalities`, `passengers_fatalities` e
        > `other_fatalities` foram **estritamente bloqueadas** do conjunto preditivo. Narrativas e laudos periciais
        > posteriores (`crash_cause` e `circumstances`) também foram eliminados, pois são produzidos somente após o término
        > da investigação da ocorrência.
        """)

        st.markdown("##### Inconsistências Históricas Tratadas no Módulo de Pré-processamento:")
        col_diag1, col_diag2, col_diag3 = st.columns(3)

        with col_diag1:
            st.markdown("""
            **1. Anos de Fabricação (`yom`)**
            - Detectados registros com `yom = 0` e valores extremos como `yom = 19567` (erros de digitação histórica).
            - *Tratamento:* Valores fora do intervalo [1900, 2026] são convertidos em `NaN` e imputados pela mediana durante o treino.
            """)

        with col_diag2:
            st.markdown("""
            **2. Tripulação a Bordo (`crew_on_board`)**
            - Detectados valores absurdos como 1.924 tripulantes (ano inserido acidentalmente no campo de contagem).
            - *Tratamento:* Contagens superiores a 50 tripulantes são saneadas e tratadas como ausentes.
            """)

        with col_diag3:
            st.markdown("""
            **3. Target com Nulos (`total_fatalities`)**
            - 26 registros no dataset original não continham informação de fatalidade.
            - *Tratamento:* Registros sem target válido são descartados antes do treinamento para não corromper o aprendizado.
            """)

else:
    st.error(f"Erro ao carregar dados: {info.get('message', 'Falha desconhecida.')}")
