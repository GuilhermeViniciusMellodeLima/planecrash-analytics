"""
Módulo de Pré-processamento e Prevenção de Data Leakage - PLANECRASH ANALYTICS
Executa a limpeza, tratamento de inconsistências, isolamento rigoroso contra vazamento de dados,
construção do target binário e montagem de pipelines de transformação com Scikit-Learn.

Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Colunas estritamente proibidas para predição por causarem Data Leakage direto ou indireto
LEAKAGE_COLUMNS = [
    "total_fatalities",        # Usada exclusivamente para derivar a classe-alvo (target)
    "crew_fatalities",         # Revela diretamente fatalidades na tripulação
    "passengers_fatalities",   # Revela diretamente fatalidades nos passageiros
    "other_fatalities",        # Revela fatalidades em solo/outros
    "survivors"                # Revela contagem pós-acidente de sobreviventes
]

# Metadados e narrativas posteriores ao acidente que não devem ser preditores
POST_CRASH_OR_META_COLUMNS = [
    "id",
    "external_id",
    "circumstances",           # Relatório narrativo redigido após a conclusão da investigação
    "crash_cause",             # Laudo pericial da causa provável emitido após o acidente
    "crash_location",          # Localização textual detalhada com ruído de texto livre
    "timestamp",               # Momento temporal específico (usado apenas em análise exploratória)
    "latitude",
    "longitude",
    "image_url",
    "hex_code",
    "registration",
    "schedule",
    "msn"
]

# Variáveis preditoras recomendadas conforme Entregável 2 e diretrizes do projeto
NUMERIC_FEATURES = [
    "crew_on_board",           # Quantidade de tripulantes a bordo
    "passengers_on_board",     # Quantidade de passageiros a bordo
    "yom",                     # Ano de fabricação da aeronave (Year of Manufacture)
    "total_on_board"           # Total de almas a bordo (Tripulação + Passageiros)
]

CATEGORICAL_FEATURES = [
    "flight_phase",            # Fase do voo (Takeoff, Flight, Landing, Taxiing, etc.)
    "aircraft",                # Modelo/tipo da aeronave
    "flight_type",             # Tipo da operação (Scheduled, Military, Training, Cargo, etc.)
    "region",                  # Região/continente do voo
    "crash_site"               # Tipo de terreno ou proximidade do aeródromo
]


def clean_and_prepare_raw_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Limpa o DataFrame inicial, remove inconsistências e cria o target e variáveis derivadas.

    Etapas:
      1. Remove registros com total_fatalities nulo.
      2. Constrói target binário: 0 (sem vítimas fatais) e 1 (com vítimas fatais).
      3. Trata outliers numéricos evidentes (erros de digitação histórica como yom=19567 ou crew=1924).
      4. Calcula a variável derivada 'total_on_board'.
    """
    df_clean = df.copy()
    initial_count = len(df_clean)

    # 1. Target binário e remoção de nulos na variável-alvo
    null_target_count = int(df_clean["total_fatalities"].isna().sum()) if "total_fatalities" in df_clean.columns else 0
    df_clean = df_clean.dropna(subset=["total_fatalities"]).copy()
    df_clean["target"] = (df_clean["total_fatalities"] > 0).astype(int)

    # 2. Tratamento de inconsistências em 'yom' (Ano de fabricação)
    # A aviação moderna opera aeronaves fabricadas entre ~1900 e os dias atuais.
    if "yom" in df_clean.columns:
        df_clean["yom"] = pd.to_numeric(df_clean["yom"], errors="coerce")
        anomalous_yom = ((df_clean["yom"] < 1900) | (df_clean["yom"] > 2026)).sum()
        df_clean.loc[(df_clean["yom"] < 1900) | (df_clean["yom"] > 2026), "yom"] = np.nan
    else:
        anomalous_yom = 0

    # 3. Tratamento de inconsistências em tripulação e passageiros
    if "crew_on_board" in df_clean.columns:
        df_clean["crew_on_board"] = pd.to_numeric(df_clean["crew_on_board"], errors="coerce")
        # Valores superiores a 50 tripulantes em registros históricos são erros de digitação (ex: 1924)
        anomalous_crew = (df_clean["crew_on_board"] > 50).sum()
        df_clean.loc[df_clean["crew_on_board"] > 50, "crew_on_board"] = np.nan
    else:
        anomalous_crew = 0

    if "passengers_on_board" in df_clean.columns:
        df_clean["passengers_on_board"] = pd.to_numeric(df_clean["passengers_on_board"], errors="coerce")
        # O maior avião de passageiros da história (Airbus A380) comporta até 853 pessoas.
        anomalous_passengers = (df_clean["passengers_on_board"] > 853).sum()
        df_clean.loc[df_clean["passengers_on_board"] > 853, "passengers_on_board"] = np.nan
    else:
        anomalous_passengers = 0

    # 4. Criação da variável derivada 'total_on_board'
    crew_clean = df_clean["crew_on_board"].fillna(0)
    passengers_clean = df_clean["passengers_on_board"].fillna(0)
    df_clean["total_on_board"] = crew_clean + passengers_clean

    stats = {
        "registros_iniciais": initial_count,
        "registros_validos": len(df_clean),
        "target_nulos_removidos": null_target_count,
        "anomalias_yom_tratadas": int(anomalous_yom),
        "anomalias_tripulacao_tratadas": int(anomalous_crew),
        "anomalias_passageiros_tratadas": int(anomalous_passengers),
        "distribuicao_target": {
            "classe_0_nao_fatais": int((df_clean["target"] == 0).sum()),
            "classe_1_fatais": int((df_clean["target"] == 1).sum()),
            "proporcao_fatais": round(float((df_clean["target"] == 1).mean()), 4)
        }
    }

    return df_clean, stats


def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Constrói a pipeline Scikit-Learn de imputação e codificação:
      - Variáveis numéricas: Imputação pela mediana + Padronização com StandardScaler.
      - Variáveis categóricas: Imputação por valor constante 'Desconhecido' + OneHotEncoder com frequência mínima.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=15, sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Realiza a separação estratificada entre treino e teste (80/20) e constrói o pré-processador.

    Garante explicitamente:
      - Nenhuma coluna proibida (leakage) entra no conjunto X.
      - O pré-processador é ajustado (fit) estritamente no conjunto de treino.
    """
    df_clean, _ = clean_and_prepare_raw_df(df)

    # Verificação de segurança contra Data Leakage
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for forbidden in LEAKAGE_COLUMNS:
        if forbidden in all_features:
            raise ValueError(f"CRÍTICO: Coluna de vazamento detectada nas features: {forbidden}")

    # Monta a matriz de features e o vetor de target
    X = df_clean[all_features].copy()
    y = df_clean["target"].copy()

    # Separação estratificada de 80% treino e 20% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = build_preprocessor_pipeline()

    return X_train, X_test, y_train, y_test, preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extrai a lista de nomes das features geradas após a codificação one-hot e transformação.
    """
    try:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                onehot_encoder = trans.named_steps["onehot"]
                encoded_cols = onehot_encoder.get_feature_names_out(cols)
                feature_names.extend(encoded_cols)
        return feature_names
    except Exception:
        return [f"feature_{i}" for i in range(100)]
