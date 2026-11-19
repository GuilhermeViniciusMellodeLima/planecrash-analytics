"""
Testes unitários para o módulo preprocessing.py - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import pandas as pd
import numpy as np
import pytest

from src.preprocessing import (
    clean_and_prepare_raw_df,
    prepare_train_test_data,
    build_preprocessor_pipeline,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


def test_clean_and_prepare_target_creation():
    """Valida a conversão correta de total_fatalities para target binário (0 e 1)."""
    mock_df = pd.DataFrame({
        "total_fatalities": [0.0, 5.0, 0.0, np.nan, 120.0],
        "crew_on_board": [2, 4, 1, 3, 10],
        "passengers_on_board": [10, 50, 0, 100, 300],
        "yom": [1980, 1995, 2005, 1970, 2018],
        "flight_phase": ["Landing", "Takeoff", "Flight", "Flight", "Approach"],
        "aircraft": ["Boeing 737", "DC-3", "A320", "AN-2", "C-130"],
        "flight_type": ["Scheduled", "Military", "Cargo", "Private", "Charter"],
        "region": ["Europe", "North America", "South America", "Asia", "Africa"],
        "crash_site": ["Airport", "Mountains", "Plain", "Sea", "Airport"]
    })

    df_clean, stats = clean_and_prepare_raw_df(mock_df)

    # Registro com NaN no target deve ter sido removido
    assert len(df_clean) == 4
    assert stats["target_nulos_removidos"] == 1

    # Verificação das classes
    assert (df_clean["target"] == 0).sum() == 2
    assert (df_clean["target"] == 1).sum() == 2


def test_absence_of_leakage_columns():
    """Garante que NENHUMA coluna de vazamento de dados ou laudo pós-fato entre no vetor X."""
    all_selected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in all_selected_features, f"VIOLAÇÃO CRÍTICA: Coluna de vazamento '{leak_col}' encontrada nas features!"


def test_treatment_of_anomalies():
    """Verifica se anos e tripulações absurdas são tratadas como nulos antes da imputação."""
    mock_df = pd.DataFrame({
        "total_fatalities": [1.0, 0.0, 1.0],
        "crew_on_board": [1924, 2, 3], # 1924 é erro de digitação histórica
        "passengers_on_board": [20, 10, 100],
        "yom": [19567, 1980, 0],       # 19567 e 0 são erros
        "flight_phase": ["Flight", "Landing", "Takeoff"],
        "aircraft": ["DC-3", "B737", "A320"],
        "flight_type": ["Military", "Scheduled", "Cargo"],
        "region": ["Europe", "Asia", "North America"],
        "crash_site": ["Mountains", "Airport", "Plain"]
    })

    df_clean, stats = clean_and_prepare_raw_df(mock_df)

    assert stats["anomalias_yom_tratadas"] == 2
    assert stats["anomalias_tripulacao_tratadas"] == 1
    assert np.isnan(df_clean.loc[0, "crew_on_board"])
    assert np.isnan(df_clean.loc[0, "yom"])
    assert np.isnan(df_clean.loc[2, "yom"])


def test_train_test_split_stratification():
    """Valida se a divisão de dados é 80% treino e 20% teste com estratificação."""
    mock_data = {
        "total_fatalities": [0.0] * 50 + [1.0] * 50,
        "crew_on_board": [2] * 100,
        "passengers_on_board": [10] * 100,
        "yom": [1990] * 100,
        "flight_phase": ["Landing"] * 100,
        "aircraft": ["DC-3"] * 100,
        "flight_type": ["Scheduled"] * 100,
        "region": ["Europe"] * 100,
        "crash_site": ["Airport"] * 100
    }
    df = pd.DataFrame(mock_data)

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(df, test_size=0.20, random_state=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    # Proporção 50/50 deve ser mantida exatamente em treino e teste devido a stratify=y
    assert y_train.mean() == 0.5
    assert y_test.mean() == 0.5
