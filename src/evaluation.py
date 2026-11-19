"""
Módulo de Avaliação e Métricas de Desempenho - PLANECRASH ANALYTICS
Calcula todas as métricas exigidas no Entregável 2 com dados reais de teste:
Acurácia, Precisão, Recall (Revocação), F1-Score, ROC-AUC e Matriz de Confusão.

Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calcula as métricas de classificação exigidas no Entregável 2.
    A classe 1 (Acidente com vítimas fatais) é tratada rigorosamente como a classe positiva.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Matriz de Confusão: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Especificidade (Taxa de verdadeiros negativos)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC e Curva ROC
    auc_val = 0.0
    fpr_list, tpr_list = [], []
    if y_prob is not None:
        try:
            auc_val = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fpr_list = fpr.tolist()
            tpr_list = tpr.tolist()
        except Exception:
            auc_val = 0.0

    return {
        "acuracia": round(acc, 4),
        "precisao": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc_val, 4),
        "especificidade": round(spec, 4),
        "matriz_confusao": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matriz": cm.tolist()
        },
        "curva_roc": {
            "fpr": fpr_list,
            "tpr": tpr_list
        }
    }


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Recebe os resultados dos modelos e gera:
      1. Tabela comparativa formatada exatamente como no Entregável 2.
      2. Diagnóstico analítico do melhor modelo com foco em Recall e F1-Score.
    """
    rows = []
    for model_name, res in results_dict.items():
        metrics = res.get("metrics", {})
        rows.append({
            "Modelo": model_name,
            "Acurácia": metrics.get("acuracia", 0.0),
            "Precisão": metrics.get("precisao", 0.0),
            "Recall": metrics.get("recall", 0.0),
            "F1-Score": metrics.get("f1_score", 0.0),
            "ROC-AUC": metrics.get("roc_auc", 0.0)
        })

    df_comp = pd.DataFrame(rows)

    # Identificação fundamentada do melhor modelo
    # Critério: Média harmônica ponderada com maior peso no Recall (identificar acidentes fatais) e F1-Score
    best_model_name = None
    best_score = -1.0
    scores = {}

    for _, row in df_comp.iterrows():
        # Score composto: 40% Recall + 30% F1-Score + 15% ROC-AUC + 15% Precisão
        comp_score = (0.40 * row["Recall"]) + (0.30 * row["F1-Score"]) + (0.15 * row["ROC-AUC"]) + (0.15 * row["Precisão"])
        scores[row["Modelo"]] = comp_score
        if comp_score > best_score:
            best_score = comp_score
            best_model_name = row["Modelo"]

    best_metrics = results_dict.get(best_model_name, {}).get("metrics", {})
    best_cm = best_metrics.get("matriz_confusao", {})

    diagnosis = {
        "melhor_modelo": best_model_name,
        "score_composto": round(best_score, 4),
        "recall_vencedor": best_metrics.get("recall", 0.0),
        "f1_vencedor": best_metrics.get("f1_score", 0.0),
        "auc_vencedor": best_metrics.get("roc_auc", 0.0),
        "falsos_negativos": best_cm.get("fn", 0),
        "falsos_positivos": best_cm.get("fp", 0),
        "explicacao_tecnica": (
            f"O algoritmo '{best_model_name}' obteve o melhor desempenho global considerando os requisitos do Entregável 2. "
            f"Alcançou um Recall de {best_metrics.get('recall', 0.0):.2%}, minimizando ocorrências de Falsos Negativos "
            f"({best_cm.get('fn', 0)} casos onde acidentes fatais foram classificados como sem vítimas), "
            f"mantendo um F1-Score de {best_metrics.get('f1_score', 0.0):.2%} e ROC-AUC de {best_metrics.get('roc_auc', 0.0):.2%}."
        )
    }

    return df_comp, diagnosis
