"""
Módulo de Visualizações e Gráficos Interativos - PLANECRASH ANALYTICS
Gera gráficos modernos com Plotly em tema escuro (azul marinho e ciano)
para análise exploratória, diagnóstico de modelos de ML e avaliação de métricas.

Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paleta de cores moderna
COLOR_CYAN = "#00D2FF"
COLOR_BLUE = "#3B82F6"
COLOR_EMERALD = "#10B981"
COLOR_ROSE = "#F43F5E"
COLOR_AMBER = "#F59E0B"
COLOR_PURPLE = "#8B5CF6"
BG_TRANSPARENT = "rgba(0,0,0,0)"
TEXT_COLOR = "#F1F5F9"
GRID_COLOR = "#334155"


def _apply_dark_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Aplica formatação dark mode padronizada em todos os gráficos Plotly."""
    fig.update_layout(
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font=dict(color=TEXT_COLOR, family="sans-serif"),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    )
    return fig


def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    """Gráfico Donut exibindo a proporção de acidentes com e sem vítimas fatais."""
    valid_df = df.dropna(subset=["total_fatalities"]).copy()
    valid_df["status"] = np.where(valid_df["total_fatalities"] > 0, "Com Vítimas Fatais (Classe 1)", "Sem Vítimas Fatais (Classe 0)")
    counts = valid_df["status"].value_counts().reset_index()
    counts.columns = ["Status", "Total"]

    fig = px.pie(
        counts,
        names="Status",
        values="Total",
        hole=0.55,
        color="Status",
        color_discrete_map={
            "Sem Vítimas Fatais (Classe 0)": COLOR_EMERALD,
            "Com Vítimas Fatais (Classe 1)": COLOR_ROSE
        }
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label+value",
        hovertemplate="<b>%{label}</b><br>Ocorrências: %{value}<br>Percentual: %{percent}<extra></extra>"
    )
    fig.update_layout(title="Distribuição da Variável-Alvo (Target)", showlegend=True)
    return _apply_dark_theme(fig, height=380)


def plot_flight_phase_distribution(df: pd.DataFrame) -> go.Figure:
    """Distribuição de ocorrências por fase do voo com discriminação de fatalidades."""
    valid_df = df.dropna(subset=["total_fatalities"]).copy()
    valid_df["fatalidade"] = np.where(valid_df["total_fatalities"] > 0, "Fatal", "Não Fatal")
    valid_df["flight_phase"] = valid_df["flight_phase"].fillna("Desconhecida")

    top_phases = valid_df["flight_phase"].value_counts().head(8).index
    filtered = valid_df[valid_df["flight_phase"].isin(top_phases)]

    grouped = filtered.groupby(["flight_phase", "fatalidade"]).size().reset_index(name="Contagem")

    fig = px.bar(
        grouped,
        x="flight_phase",
        y="Contagem",
        color="fatalidade",
        barmode="group",
        color_discrete_map={"Não Fatal": COLOR_EMERALD, "Fatal": COLOR_ROSE},
        labels={"flight_phase": "Fase do Voo", "Contagem": "Número de Ocorrências", "fatalidade": "Gravidade"}
    )
    fig.update_layout(title="Acidentes por Fase de Voo e Gravidade", xaxis_tickangle=-25)
    return _apply_dark_theme(fig, height=420)


def plot_flight_type_distribution(df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Distribuição por tipo de voo/operação."""
    valid_df = df.dropna(subset=["total_fatalities"]).copy()
    valid_df["flight_type"] = valid_df["flight_type"].fillna("Não informado")
    top_types = valid_df["flight_type"].value_counts().head(top_n).index

    filtered = valid_df[valid_df["flight_type"].isin(top_types)]
    summary = filtered.groupby("flight_type").agg(
        Total=("id", "count"),
        Fatais=("total_fatalities", lambda s: (s > 0).sum())
    ).reset_index()
    summary["Taxa_Fatalidade"] = (summary["Fatais"] / summary["Total"]) * 100
    summary = summary.sort_values(by="Total", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=summary["flight_type"],
        x=summary["Total"],
        orientation="h",
        name="Total de Ocorrências",
        marker_color=COLOR_BLUE
    ))
    fig.add_trace(go.Bar(
        y=summary["flight_type"],
        x=summary["Fatais"],
        orientation="h",
        name="Ocorrências Fatais",
        marker_color=COLOR_ROSE
    ))
    fig.update_layout(
        title=f"Top {top_n} Tipos de Operação Aérea",
        barmode="overlay",
        xaxis_title="Quantidade de Acidentes",
        yaxis_title="Tipo de Operação"
    )
    return _apply_dark_theme(fig, height=450)


def plot_geographic_distribution(df: pd.DataFrame) -> go.Figure:
    """Distribuição geográfica dos incidentes por região/continente."""
    valid_df = df.copy()
    valid_df["region"] = valid_df["region"].fillna("Desconhecida")
    counts = valid_df["region"].value_counts().reset_index()
    counts.columns = ["Região", "Ocorrências"]

    fig = px.bar(
        counts,
        x="Região",
        y="Ocorrências",
        text="Ocorrências",
        color="Ocorrências",
        color_continuous_scale="Viridis",
        labels={"Região": "Região Geográfica", "Ocorrências": "Total de Acidentes"}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Distribuição Geográfica por Região / Continente", xaxis_tickangle=-25)
    return _apply_dark_theme(fig, height=420)


def plot_people_on_board_vs_fatalities(df: pd.DataFrame) -> go.Figure:
    """Relação entre pessoas a bordo (tripulação + passageiros) e ocorrência de fatalidades."""
    valid_df = df.dropna(subset=["total_fatalities"]).copy()
    valid_df["crew"] = pd.to_numeric(valid_df["crew_on_board"], errors="coerce").fillna(0)
    valid_df["pass"] = pd.to_numeric(valid_df["passengers_on_board"], errors="coerce").fillna(0)
    valid_df["total_on_board"] = valid_df["crew"] + valid_df["pass"]

    # Filtra outliers extremos para melhor visualização
    filtered = valid_df[(valid_df["total_on_board"] > 0) & (valid_df["total_on_board"] <= 150)].copy()
    filtered["Gravidade"] = np.where(filtered["total_fatalities"] > 0, "Com Fatais (Classe 1)", "Sem Fatais (Classe 0)")

    fig = px.histogram(
        filtered,
        x="total_on_board",
        color="Gravidade",
        barmode="overlay",
        nbins=40,
        color_discrete_map={"Sem Fatais (Classe 0)": COLOR_EMERALD, "Com Fatais (Classe 1)": COLOR_ROSE},
        labels={"total_on_board": "Total de Pessoas a Bordo", "count": "Frequência"}
    )
    fig.update_layout(title="Distribuição de Pessoas a Bordo por Gravidade do Incidente")
    return _apply_dark_theme(fig, height=400)


def plot_accidents_over_time(df: pd.DataFrame) -> Optional[go.Figure]:
    """Série histórica anual de acidentes aéreos com base no campo timestamp."""
    if "timestamp" not in df.columns:
        return None
    try:
        temp_df = df.dropna(subset=["timestamp"]).copy()
        temp_df["ano"] = pd.to_datetime(temp_df["timestamp"], errors="coerce").dt.year
        temp_df = temp_df.dropna(subset=["ano"])
        temp_df["ano"] = temp_df["ano"].astype(int)
        temp_df = temp_df[(temp_df["ano"] >= 1910) & (temp_df["ano"] <= 2026)]

        grouped = temp_df.groupby("ano").agg(
            Total=("id", "count"),
            Fatais=("total_fatalities", lambda s: (s > 0).sum())
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grouped["ano"],
            y=grouped["Total"],
            mode="lines",
            name="Total de Acidentes",
            line=dict(color=COLOR_BLUE, width=2)
        ))
        fig.add_trace(go.Scatter(
            x=grouped["ano"],
            y=grouped["Fatais"],
            mode="lines",
            name="Acidentes com Vítimas Fatais",
            line=dict(color=COLOR_ROSE, width=2)
        ))
        fig.update_layout(
            title="Evolução Histórica Anual de Acidentes Aeronáuticos",
            xaxis_title="Ano da Ocorrência",
            yaxis_title="Número de Incidentes",
            hovermode="x unified"
        )
        return _apply_dark_theme(fig, height=400)
    except Exception:
        return None


def plot_metrics_comparison(df_comp: pd.DataFrame) -> go.Figure:
    """Gráfico de barras agrupadas comparando todas as métricas dos três modelos."""
    metrics_cols = ["Acurácia", "Precisão", "Recall", "F1-Score", "ROC-AUC"]
    melted = df_comp.melt(id_vars=["Modelo"], value_vars=metrics_cols, var_name="Métrica", value_name="Pontuação")

    fig = px.bar(
        melted,
        x="Métrica",
        y="Pontuação",
        color="Modelo",
        barmode="group",
        text_auto=".2%",
        color_discrete_sequence=[COLOR_BLUE, COLOR_CYAN, COLOR_EMERALD]
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="Comparativo de Métricas de Desempenho no Conjunto de Teste (20%)",
        yaxis_range=[0, 1.1],
        yaxis_tickformat=".0%"
    )
    return _apply_dark_theme(fig, height=450)


def plot_individual_metric_bar(df_comp: pd.DataFrame, metric_col: str) -> go.Figure:
    """Gráfico individual para uma métrica específica (ex: Recall ou F1-Score)."""
    fig = px.bar(
        df_comp,
        x="Modelo",
        y=metric_col,
        color="Modelo",
        text_auto=".2%",
        color_discrete_sequence=[COLOR_BLUE, COLOR_CYAN, COLOR_EMERALD]
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=f"Comparativo Específico: {metric_col}",
        yaxis_range=[0, 1.05],
        yaxis_tickformat=".0%"
    )
    return _apply_dark_theme(fig, height=350)


def plot_confusion_matrix_heatmap(cm_dict: Dict[str, Any], title: str) -> go.Figure:
    """Matriz de Confusão formatada com contagens absolutas e percentuais por célula."""
    tn = cm_dict.get("tn", 0)
    fp = cm_dict.get("fp", 0)
    fn = cm_dict.get("fn", 0)
    tp = cm_dict.get("tp", 0)

    z = [[tn, fp], [fn, tp]]
    total = tn + fp + fn + tp

    text = [
        [f"Verdadeiro Negativo<br><b>{tn}</b> ({tn/total:.1%})", f"Falso Positivo<br><b>{fp}</b> ({fp/total:.1%})"],
        [f"Falso Negativo (Crítico!)<br><b>{fn}</b> ({fn/total:.1%})", f"Verdadeiro Positivo<br><b>{tp}</b> ({tp/total:.1%})"]
    ]

    fig = ff_cm = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predito: Sem Vítimas (0)", "Predito: Com Vítimas (1)"],
        y=["Real: Sem Vítimas (0)", "Real: Com Vítimas (1)"],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"},
        colorscale="Blues",
        showscale=False
    ))

    fig.update_layout(
        title=f"Matriz de Confusão — {title}",
        xaxis_title="Previsão do Algoritmo",
        yaxis_title="Realidade Histórica",
        yaxis_autorange="reversed"
    )
    return _apply_dark_theme(fig, height=380)


def plot_combined_roc_curves(results_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Compara as curvas ROC de todos os modelos em um único plano cartesiano."""
    fig = go.Figure()
    colors = [COLOR_BLUE, COLOR_CYAN, COLOR_EMERALD]

    for idx, (name, res) in enumerate(results_dict.items()):
        metrics = res.get("metrics", {})
        roc_data = metrics.get("curva_roc", {})
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        auc_val = metrics.get("roc_auc", 0.0)

        if fpr and tpr:
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC = {auc_val:.3f})",
                line=dict(color=colors[idx % len(colors)], width=2.5)
            ))

    # Linha diagonal de classificação aleatória
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Classificador Aleatório (AUC = 0.50)",
        line=dict(color="#64748B", dash="dash")
    ))

    fig.update_layout(
        title="Curvas ROC Comparativas (Receiver Operating Characteristic)",
        xaxis_title="Taxa de Falsos Positivos (1 - Especificidade)",
        yaxis_title="Taxa de Verdadeiros Positivos (Recall / Sensibilidade)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05])
    )
    return _apply_dark_theme(fig, height=450)


def plot_feature_importances(results_dict: Dict[str, Dict[str, Any]], model_name: str = "Random Forest") -> Optional[go.Figure]:
    """Exibe o ranking das variáveis de maior importância para o modelo selecionado."""
    model_data = results_dict.get(model_name, {})
    imp_dict = model_data.get("feature_importance", {})

    if not imp_dict:
        return None

    items = sorted(imp_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    features = [it[0].replace("cat__", "").replace("num__", "") for it in items]
    values = [it[1] for it in items]

    fig = px.bar(
        x=values[::-1],
        y=features[::-1],
        orientation="h",
        color=values[::-1],
        color_continuous_scale="Teal",
        labels={"x": "Grau de Importância / Peso", "y": "Variável / Característica"}
    )
    fig.update_layout(
        title=f"Importância das Variáveis — {model_name}",
        coloraxis_showscale=False
    )
    return _apply_dark_theme(fig, height=460)
