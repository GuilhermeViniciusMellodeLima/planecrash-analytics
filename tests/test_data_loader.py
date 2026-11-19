"""
Testes unitários para o módulo data_loader.py - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import os
import pandas as pd
import pytest
from src.data_loader import load_raw_data, get_dataset_metadata, DEFAULT_LOCAL_PATH


def test_load_raw_data_local():
    """Verifica se o carregamento a partir do cache local CSV funciona corretamente."""
    assert os.path.exists(DEFAULT_LOCAL_PATH), "Arquivo local CSV deve existir para execução dos testes."
    df, info = load_raw_data(source_preference="local")

    assert df is not None, "DataFrame não deve ser None."
    assert not df.empty, "DataFrame não deve estar vazio."
    assert info["success"] is True, "Status do carregamento deve ser sucesso."
    assert info["rows"] > 20000, "Dataset PlaneCrashData deve conter mais de 20.000 registros."
    assert "total_fatalities" in df.columns, "Coluna total_fatalities deve estar presente."


def test_get_dataset_metadata():
    """Testa a geração de metadados estatísticos e contagem de nulos."""
    df, _ = load_raw_data(source_preference="local")
    meta = get_dataset_metadata(df)

    assert "total_registros" in meta
    assert "total_colunas" in meta
    assert "colunas_detalhes" in meta
    assert "target_summary" in meta

    t_sum = meta["target_summary"]
    assert t_sum["classe_0_nao_fatais"] > 0, "Deve haver ocorrências da Classe 0."
    assert t_sum["classe_1_fatais"] > 0, "Deve haver ocorrências da Classe 1."
    assert t_sum["pct_fatais"] > 40 and t_sum["pct_fatais"] < 70, "Proporção de fatais deve ser condizente."
