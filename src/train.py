"""
Módulo de Treinamento e Persistência de Modelos - PLANECRASH ANALYTICS
Treina os três algoritmos supervisionados obrigatórios do Entregável 2:
  1. Regressão Logística (Baseline)
  2. Árvore de Decisão
  3. Random Forest

Garante persistência via Joblib e salvamento de métricas reais em JSON.

Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import os
import json
import joblib
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import prepare_train_test_data, get_feature_names
from src.evaluation import compute_classification_metrics, compare_models

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_default_models(use_class_weight: bool = True) -> Dict[str, Any]:
    """
    Retorna instâncias dos três modelos obrigatórios com hiperparâmetros balanceados e reprodutíveis.
    """
    weight = "balanced" if use_class_weight else None

    return {
        "Regressão Logística": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=weight,
            solver="lbfgs"
        ),
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight=weight
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=15,
            min_samples_leaf=5,
            random_state=42,
            class_weight=weight,
            n_jobs=-1
        )
    }


def train_all_models(
    df: pd.DataFrame,
    use_class_weight: bool = True,
    save_on_disk: bool = True
) -> Dict[str, Any]:
    """
    Executa o ciclo completo de treinamento dos três modelos:
      1. Divisão estratificada 80/20.
      2. Ajuste do pré-processador estritamente em X_train.
      3. Treinamento de cada modelo via Pipeline Scikit-Learn.
      4. Predição e cálculo de probabilidades em X_test.
      5. Cálculo de métricas completas com compute_classification_metrics.
      6. Extração de importâncias/coeficientes das variáveis.
      7. Diagnóstico comparativo.
      8. Persistência em disco via Joblib.
    """
    # 1. Preparação dos dados
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(df)

    # 2. Fit do pré-processador apenas no conjunto de treino
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    raw_models = get_default_models(use_class_weight=use_class_weight)
    results = {}

    for name, clf in raw_models.items():
        # Treina o classificador nos dados transformados
        clf.fit(X_train_trans, y_train)

        # Predições no conjunto de teste
        y_pred = clf.predict(X_test_trans)
        y_prob = clf.predict_proba(X_test_trans)[:, 1] if hasattr(clf, "predict_proba") else None

        # Predições no conjunto de treino (para diagnóstico de overfitting)
        y_train_pred = clf.predict(X_train_trans)

        metrics_test = compute_classification_metrics(y_test, y_pred, y_prob)
        metrics_train = compute_classification_metrics(y_train, y_train_pred)

        # Extração de relevância das variáveis
        importance_dict = {}
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            # Mapeia para as 15 variáveis de maior peso
            indices = np.argsort(importances)[::-1][:15]
            importance_dict = {
                feature_names[i]: round(float(importances[i]), 4)
                for i in indices if i < len(feature_names)
            }
        elif hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
            indices = np.argsort(np.abs(coefs))[::-1][:15]
            importance_dict = {
                feature_names[i]: round(float(coefs[i]), 4)
                for i in indices if i < len(feature_names)
            }

        # Cria pipeline completo para inferência em produção
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        results[name] = {
            "model_obj": clf,
            "pipeline": full_pipeline,
            "metrics": metrics_test,
            "train_metrics": metrics_train,
            "feature_importance": importance_dict
        }

    # Gera tabela comparativa e diagnóstico
    df_comp, diagnosis = compare_models(results)

    payload = {
        "results": results,
        "preprocessor": preprocessor,
        "comparison_table": df_comp,
        "diagnosis": diagnosis,
        "feature_names": feature_names,
        "metadata": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "use_class_weight": use_class_weight,
            "train_target_distribution": {
                "classe_0": int((y_train == 0).sum()),
                "classe_1": int((y_train == 1).sum())
            },
            "test_target_distribution": {
                "classe_0": int((y_test == 0).sum()),
                "classe_1": int((y_test == 1).sum())
            }
        }
    }

    if save_on_disk:
        save_artifacts(payload)

    return payload


def _model_to_filename(model_name: str) -> str:
    """Converte nome do modelo para nome de arquivo seguro e padronizado sem acentuação."""
    slug = model_name.lower().strip()
    replacements = {
        "ã": "a", "á": "a", "à": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u",
        " ": "_"
    }
    for char, rep in replacements.items():
        slug = slug.replace(char, rep)
    return f"{slug}.joblib"


def save_artifacts(payload: Dict[str, Any], output_dir: str = MODELS_DIR) -> None:
    """
    Persiste os modelos treinados, o pré-processador e as métricas em formato serializado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Salva o pré-processador isolado
    joblib.dump(payload["preprocessor"], os.path.join(output_dir, "preprocessor.joblib"))

    # 2. Salva cada pipeline de modelo com nome normalizado
    for name, data in payload["results"].items():
        filename = _model_to_filename(name)
        joblib.dump(data["pipeline"], os.path.join(output_dir, filename))

    # 3. Serializa as métricas em JSON legível
    metrics_export = {
        "metadata": payload["metadata"],
        "diagnosis": payload["diagnosis"],
        "comparison_table": payload["comparison_table"].to_dict(orient="records"),
        "models": {
            name: {
                "metrics": data["metrics"],
                "train_metrics": data["train_metrics"],
                "feature_importance": data["feature_importance"]
            }
            for name, data in payload["results"].items()
        }
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2, ensure_ascii=False)


def are_models_trained(models_dir: str = MODELS_DIR) -> bool:
    """
    Verifica se os arquivos dos três modelos treinados e do pré-processador existem em disco.
    """
    expected_files = [
        "preprocessor.joblib",
        "regressao_logistica.joblib",
        "arvore_de_decisao.joblib",
        "random_forest.joblib",
        "metrics.json"
    ]
    return all(os.path.exists(os.path.join(models_dir, f)) for f in expected_files)


def load_saved_artifacts(models_dir: str = MODELS_DIR) -> Optional[Dict[str, Any]]:
    """
    Carrega os modelos treinados e o relatório de métricas a partir do disco sem necessidade de retreinar.
    """
    if not are_models_trained(models_dir):
        return None

    try:
        preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
        pipelines = {
            "Regressão Logística": joblib.load(os.path.join(models_dir, "regressao_logistica.joblib")),
            "Árvore de Decisão": joblib.load(os.path.join(models_dir, "arvore_de_decisao.joblib")),
            "Random Forest": joblib.load(os.path.join(models_dir, "random_forest.joblib"))
        }

        with open(os.path.join(models_dir, "metrics.json"), "r", encoding="utf-8") as f:
            metrics_export = json.load(f)

        df_comp = pd.DataFrame(metrics_export.get("comparison_table", []))

        results = {}
        for name, pipe in pipelines.items():
            model_info = metrics_export.get("models", {}).get(name, {})
            results[name] = {
                "pipeline": pipe,
                "metrics": model_info.get("metrics", {}),
                "train_metrics": model_info.get("train_metrics", {}),
                "feature_importance": model_info.get("feature_importance", {})
            }

        return {
            "results": results,
            "preprocessor": preprocessor,
            "comparison_table": df_comp,
            "diagnosis": metrics_export.get("diagnosis", {}),
            "metadata": metrics_export.get("metadata", {})
        }
    except Exception as e:
        print(f"Erro ao carregar modelos salvos: {e}")
        return None
