"""
Módulo de Predição e Inferência - PLANECRASH ANALYTICS
Executa previsões reais a partir das características de um incidente aéreo fornecidas pelo usuário,
utilizando os pipelines completos de Machine Learning previamente treinados.

Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def predict_incident_severity(
    pipeline: Any,
    input_data: Dict[str, Any],
    model_name: str = "Random Forest"
) -> Dict[str, Any]:
    """
    Realiza a predição para um novo evento aéreo com base nas variáveis informadas.

    Args:
        pipeline: Pipeline Scikit-Learn treinado (composto por preprocessor + classifier).
        input_data: Dicionário contendo os valores para as features obrigatórias.
        model_name: Nome descritivo do modelo escolhido.

    Returns:
        Dicionário com a classe predita, probabilidades, nível de risco e explicação contextual.
    """
    # 1. Validação e saneamento dos campos
    crew = float(max(0, input_data.get("crew_on_board", 2)))
    passengers = float(max(0, input_data.get("passengers_on_board", 0)))
    yom = float(input_data.get("yom", 1980))
    flight_phase = str(input_data.get("flight_phase", "Landing (descent or approach)"))
    aircraft = str(input_data.get("aircraft", "Desconhecido"))
    flight_type = str(input_data.get("flight_type", "Scheduled Revenue Flight"))
    region = str(input_data.get("region", "South America"))
    crash_site = str(input_data.get("crash_site", "Airport (less than 10 km from airport)"))

    # 2. Construção do DataFrame de inferência com a variável derivada
    df_input = pd.DataFrame([{
        "flight_phase": flight_phase,
        "aircraft": aircraft,
        "flight_type": flight_type,
        "region": region,
        "crash_site": crash_site,
        "crew_on_board": crew,
        "passengers_on_board": passengers,
        "yom": yom,
        "total_on_board": crew + passengers
    }])

    # 3. Execução da predição através do pipeline
    try:
        pred_class = int(pipeline.predict(df_input)[0])

        prob_fatal = 0.5
        prob_non_fatal = 0.5
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df_input)[0]
            prob_non_fatal = float(probs[0])
            prob_fatal = float(probs[1])
        else:
            # Para modelos sem predict_proba nativo
            prob_fatal = 1.0 if pred_class == 1 else 0.0
            prob_non_fatal = 1.0 - prob_fatal

    except Exception as e:
        return {
            "success": False,
            "error": f"Erro durante a inferência do modelo: {str(e)}"
        }

    # 4. Geração do diagnóstico descritivo e interpretação
    if pred_class == 1:
        resultado_rotulo = "Classe 1 — Acidente com Vítimas Fatais"
        descricao_curta = "O modelo identificou alta probabilidade estatística de fatalidades para este perfil de ocorrência."
        badge_cor = "danger"
    else:
        resultado_rotulo = "Classe 0 — Acidente sem Vítimas Fatais"
        descricao_curta = "O modelo estimou predomínio estatístico de sobrevivência total para os parâmetros informados."
        badge_cor = "success"

    # Fatores de risco identificados no perfil
    fatores_relevantes = []
    if "Flight" in flight_phase or "En route" in flight_phase:
        fatores_relevantes.append("Fase de voo em rota/cruzeiro (historicamente associada a maior energia de impacto).")
    elif "Landing" in flight_phase or "Takeoff" in flight_phase:
        fatores_relevantes.append("Fase crítica de aproximação ou decolagem.")

    if "Mountains" in crash_site or "Lake" in crash_site or "Sea" in crash_site:
        fatores_relevantes.append("Terreno hostil ou aquático (dificulta resgate imediato e aumenta severidade).")
    elif "Airport" in crash_site:
        fatores_relevantes.append("Proximidade de aeródromo (disponibilidade de infraestrutura de salvamento e combate a incêndio).")

    if (crew + passengers) > 100:
        fatores_relevantes.append("Elevado número de ocupantes a bordo (aeronave de grande porte).")

    return {
        "success": True,
        "modelo_utilizado": model_name,
        "classe_predita": pred_class,
        "rotulo_classe": resultado_rotulo,
        "descricao_curta": descricao_curta,
        "badge_cor": badge_cor,
        "probabilidade_fatal": round(prob_fatal * 100, 2),
        "probabilidade_nao_fatal": round(prob_non_fatal * 100, 2),
        "fatores_analisados": fatores_relevantes,
        "parametros_recebidos": {
            "aeronave": aircraft,
            "fase_voo": flight_phase,
            "tipo_voo": flight_type,
            "regiao": region,
            "local_queda": crash_site,
            "tripulacao": int(crew),
            "passageiros": int(passengers),
            "total_a_bordo": int(crew + passengers),
            "ano_fabricacao": int(yom)
        },
        "disclaimer_academico": (
            "AVISO ÉTICO E METODOLÓGICO: Esta predição reflete exclusivamente padrões estatísticos "
            "encontrados no dataset histórico PlaneCrashData. Não constitui laudo aeronáutico, "
            "não possui certificação da ICAO/FAA/CENIPA e não deve ser utilizada para operação de voo real."
        )
    }
