"""
Testes unitários para o treinamento, avaliação e predição dos modelos - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import os
import pandas as pd
import pytest

from src.data_loader import load_raw_data
from src.train import train_all_models, are_models_trained, load_saved_artifacts
from src.evaluation import compute_classification_metrics
from src.prediction import predict_incident_severity


@pytest.fixture(scope="session")
def trained_artifacts():
    """Garante que os modelos estejam treinados e disponíveis para a suíte de testes."""
    if not are_models_trained():
        df, _ = load_raw_data(source_preference="local")
        payload = train_all_models(df, use_class_weight=True, save_on_disk=True)
        return payload
    return load_saved_artifacts()


def test_models_exist_and_trained(trained_artifacts):
    """Verifica se os três modelos obrigatórios foram treinados e estruturados."""
    results = trained_artifacts.get("results", {})
    expected_models = ["Regressão Logística", "Árvore de Decisão", "Random Forest"]

    for model_name in expected_models:
        assert model_name in results, f"Modelo obrigatório '{model_name}' não encontrado nos resultados."
        assert "pipeline" in results[model_name]
        assert "metrics" in results[model_name]


def test_metrics_integrity(trained_artifacts):
    """Garante que todas as 5 métricas obrigatórias do Entregável 2 foram calculadas corretamente."""
    results = trained_artifacts.get("results", {})

    for model_name, data in results.items():
        m = data["metrics"]
        assert 0.0 <= m["acuracia"] <= 1.0, f"Acurácia fora do intervalo válido em {model_name}"
        assert 0.0 <= m["precisao"] <= 1.0, f"Precisão fora do intervalo válido em {model_name}"
        assert 0.0 <= m["recall"] <= 1.0, f"Recall fora do intervalo válido em {model_name}"
        assert 0.0 <= m["f1_score"] <= 1.0, f"F1-Score fora do intervalo válido em {model_name}"
        assert 0.0 <= m["roc_auc"] <= 1.0, f"ROC-AUC fora do intervalo válido em {model_name}"

        # Matriz de confusão
        cm = m["matriz_confusao"]
        total_cm = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        assert total_cm > 5000, "Matriz de confusão deve cobrir o conjunto de teste de ~5.515 amostras."


def test_prediction_inference(trained_artifacts):
    """Testa o funcionamento da inferência com dados de exemplo."""
    results = trained_artifacts.get("results", {})
    rf_pipe = results["Random Forest"]["pipeline"]

    sample_input = {
        "flight_phase": "Flight",
        "aircraft": "Boeing 737",
        "flight_type": "Scheduled Revenue Flight",
        "region": "South America",
        "crash_site": "Mountains",
        "crew_on_board": 4,
        "passengers_on_board": 120,
        "yom": 1998
    }

    pred = predict_incident_severity(rf_pipe, sample_input, model_name="Random Forest")

    assert pred["success"] is True
    assert pred["classe_predita"] in [0, 1]
    assert 0.0 <= pred["probabilidade_fatal"] <= 100.0
    assert 0.0 <= pred["probabilidade_nao_fatal"] <= 100.0
    assert round(pred["probabilidade_fatal"] + pred["probabilidade_nao_fatal"], 1) == 100.0
    assert "disclaimer_academico" in pred
