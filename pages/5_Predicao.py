"""
Página: Sistema de Predição em Tempo Real - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st
import pandas as pd

from src.ui_components import inject_custom_css, render_header, render_sidebar_info, render_metric_card
from src.train import are_models_trained, load_saved_artifacts
from src.prediction import predict_incident_severity

st.set_page_config(
    page_title="PLANECRASH ANALYTICS — Predição",
    page_icon="🎯",
    layout="wide"
)

inject_custom_css()
render_sidebar_info()

render_header(
    title="Simulador de Classificação de Gravidade",
    subtitle="Previsão Real Supervisionada da Ocorrência de Vítimas Fatais com Base em Parâmetros de Voo"
)

# Validação do estado dos modelos
if not are_models_trained():
    st.warning("⚠️ Os modelos de Machine Learning precisam ser treinados antes de realizar previsões.")
    if st.button("Ir para Treinamento dos Modelos"):
        st.switch_page("pages/3_Treinamento.py")
    st.stop()

artifacts = load_saved_artifacts()
if not artifacts:
    st.error("Erro ao carregar pipelines treinados.")
    st.stop()

results = artifacts.get("results", {})
model_choices = list(results.keys())

# Lista de opções refinadas com base no dataset real
PHASES = [
    "Landing (descent or approach)",
    "Flight",
    "Takeoff (climb)",
    "Taxiing",
    "Parking",
    "Approach",
    "En route",
    "Initial climb"
]

AIRCRAFTS = [
    "Douglas DC-3",
    "Boeing 737",
    "Douglas C-47 Skytrain (DC-3)",
    "PZL-Mielec AN-2",
    "Curtiss C-46 Commando",
    "Lockheed C-130 Hercules",
    "Cessna 172",
    "Cessna 208 Caravan",
    "De Havilland DHC-6 Twin Otter",
    "Airbus A320",
    "Piper PA-31 Navajo",
    "Beechcraft King Air",
    "Embraer EMB-110 Bandeirante",
    "Boeing 747",
    "Outro / Não listado"
]

TYPES = [
    "Scheduled Revenue Flight",
    "Military",
    "Training",
    "Cargo",
    "Private",
    "Charter/Taxi (Non Scheduled Revenue Flight)",
    "Executive/Corporate/Business",
    "Survey / Patrol / Reconnaissance",
    "Positioning"
]

REGIONS = [
    "South America",
    "North America",
    "Europe",
    "Asia",
    "Africa",
    "Oceania",
    "Central America"
]

CRASH_SITES = [
    "Airport (less than 10 km from airport)",
    "Plain, Valley",
    "Mountains",
    "Lake, Sea, Ocean, River",
    "City",
    "Desert"
]

st.markdown("""
<div style="background: #131E33; border: 1px solid #1E293B; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">
    <p style="color: #94A3B8; font-size: 0.92rem; margin: 0;">
        Preencha os parâmetros pré-acidente da ocorrência e escolha o algoritmo de Machine Learning.
        O sistema processará as variáveis pelo pipeline calibrado (imputação, escalonamento e OneHotEncoding)
        e estimará a classe de gravidade e as respectivas probabilidades.
    </p>
</div>
""", unsafe_allow_html=True)

with st.form("form_predicao"):
    st.markdown("#### 1. Seleção do Modelo Preditivo")
    chosen_model = st.selectbox(
        "Selecione o modelo treinado para inferência:",
        options=model_choices,
        index=len(model_choices) - 1, # Padrão: Random Forest
        help="O Random Forest apresentou o maior Recall e F1-Score no conjunto de teste."
    )

    st.markdown("#### 2. Características Operacionais da Aeronave e do Voo")
    col1, col2, col3 = st.columns(3)

    with col1:
        sel_phase = st.selectbox("Fase do Voo (`flight_phase`):", options=PHASES, index=0)
        sel_aircraft = st.selectbox("Modelo da Aeronave (`aircraft`):", options=AIRCRAFTS, index=0)
        if sel_aircraft == "Outro / Não listado":
            custom_aircraft = st.text_input("Especifique o modelo:", value="Desconhecido")
        else:
            custom_aircraft = sel_aircraft

    with col2:
        sel_type = st.selectbox("Tipo de Operação (`flight_type`):", options=TYPES, index=0)
        sel_region = st.selectbox("Região Geográfica (`region`):", options=REGIONS, index=0)
        sel_site = st.selectbox("Terreno / Localização (`crash_site`):", options=CRASH_SITES, index=0)

    with col3:
        inp_crew = st.number_input("Tripulantes a Bordo (`crew_on_board`):", min_value=0, max_value=40, value=3, step=1)
        inp_pass = st.number_input("Passageiros a Bordo (`passengers_on_board`):", min_value=0, max_value=850, value=25, step=1)
        inp_yom = st.number_input("Ano de Fabricação da Aeronave (`yom`):", min_value=1910, max_value=2026, value=1978, step=1)

    submit_button = st.form_submit_button("⚡ Realizar Previsão de Gravidade", type="primary", use_container_width=True)

if submit_button:
    chosen_pipeline = results[chosen_model]["pipeline"]

    input_payload = {
        "flight_phase": sel_phase,
        "aircraft": custom_aircraft,
        "flight_type": sel_type,
        "region": sel_region,
        "crash_site": sel_site,
        "crew_on_board": inp_crew,
        "passengers_on_board": inp_pass,
        "yom": inp_yom
    }

    with st.spinner("Executando inferência pelo pipeline Scikit-Learn..."):
        pred_res = predict_incident_severity(chosen_pipeline, input_payload, model_name=chosen_model)

    if pred_res.get("success", False):
        st.markdown("---")
        st.markdown("### 📋 Resultado da Classificação")

        is_fatal = (pred_res["classe_predita"] == 1)
        badge_bg = "rgba(244, 63, 94, 0.2)" if is_fatal else "rgba(16, 185, 129, 0.2)"
        badge_border = "#F43F5E" if is_fatal else "#10B981"
        badge_text_color = "#FB7185" if is_fatal else "#34D399"

        st.markdown(f"""
        <div style="background: {badge_bg}; border: 2px solid {badge_border}; border-radius: 14px; padding: 1.6rem 2rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: {badge_text_color}; margin-bottom: 4px;">
                Previsão do Algoritmo: {pred_res['modelo_utilizado']}
            </div>
            <h2 style="color: #F8FAFC; margin: 0 0 8px 0; font-size: 1.8rem;">
                {pred_res['rotulo_classe']}
            </h2>
            <div style="color: #CBD5E1; font-size: 0.96rem; line-height: 1.5;">
                {pred_res['descricao_curta']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilidades Estimadas
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            render_metric_card(
                label="Probabilidade: Com Vítimas Fatais (Classe 1)",
                value=f"{pred_res['probabilidade_fatal']}%",
                hint="Risco estatístico estimado",
                border_color=badge_border if is_fatal else "#334155"
            )
        with res_col2:
            render_metric_card(
                label="Probabilidade: Sem Vítimas Fatais (Classe 0)",
                value=f"{pred_res['probabilidade_nao_fatal']}%",
                hint="Chance de sobrevivência total estimada",
                border_color=badge_border if not is_fatal else "#334155"
            )

        # Barra comparativa de probabilidades
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(pred_res['probabilidade_fatal'] / 100.0)

        # Fatores e Justificativa
        st.markdown("#### 🔍 Fatores Determinantes Identificados no Perfil:")
        fatores = pred_res.get("fatores_analisados", [])
        if fatores:
            for f in fatores:
                st.markdown(f"- ✈️ {f}")
        else:
            st.markdown("- Perfil operacional típico sem anomalias extremas identificadas.")

        # Aviso Ético Obrigatório
        st.markdown("""
        > [!CAUTION]
        > **Aviso Ético e Limitação Metodológica:**
        > Essa previsão é uma classificação estatística baseada nos padrões históricos do dataset acadêmico *PlaneCrashData*.
        > **Não deve ser apresentada como certeza, diagnóstico de segurança ou previsão confiável de um acidente futuro.**
        > A aviação real conta com rigorosos procedimentos de segurança, engenharia de confiabilidade e investigação pericial
        > (CENIPA, NTSB, ICAO) que não podem ser substituídos por modelos supervisionados de domínio público.
        """)

    else:
        st.error(f"Erro ao processar predição: {pred_res.get('error', 'Desconhecido')}")
