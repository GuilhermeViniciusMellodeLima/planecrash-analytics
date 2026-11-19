"""
Módulo de Carregamento de Dados - PLANECRASH ANALYTICS
Responsável por obter o dataset PlaneCrashData (remoto via Hugging Face ou local via CSV),
validar a integridade estrutural e fornecer metadados detalhados para a aplicação.

Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd

# Caminho padrão para cache local do dataset
DEFAULT_LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "planecrash_raw.csv")
HF_DATASET_NAME = "MauroPello/planecrashdata"


def load_raw_data(
    source_preference: str = "auto",
    local_path: Optional[str] = None,
    uploaded_file: Any = None
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Carrega o dataset PlaneCrashData a partir de Hugging Face, arquivo local ou arquivo enviado pelo usuário.

    Args:
        source_preference: 'auto', 'huggingface' ou 'local'.
        local_path: Caminho opcional para arquivo CSV local.
        uploaded_file: Objeto de arquivo carregado via interface (ex: st.file_uploader).

    Returns:
        Tuple contendo o DataFrame (ou None em caso de falha crítica) e um dicionário com metadados do carregamento.
    """
    path_to_use = local_path or DEFAULT_LOCAL_PATH
    info: Dict[str, Any] = {
        "source_used": None,
        "success": False,
        "rows": 0,
        "cols": 0,
        "message": "",
        "error_details": None
    }

    # 1. Caso o usuário tenha feito upload direto de um CSV
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            info["source_used"] = "Arquivo enviado pelo usuário"
            info["success"] = True
            info["rows"] = len(df)
            info["cols"] = len(df.columns)
            info["message"] = f"Arquivo CSV carregado com sucesso via upload ({info['rows']} registros, {info['cols']} colunas)."
            return df, info
        except Exception as e:
            info["error_details"] = str(e)
            info["message"] = f"Falha ao processar arquivo enviado: {e}"
            return None, info

    # 2. Tentativa remota via Hugging Face (se preferência for 'auto' ou 'huggingface')
    if source_preference in ("auto", "huggingface"):
        try:
            from datasets import load_dataset
            ds = load_dataset(HF_DATASET_NAME)
            split_name = list(ds.keys())[0]
            df = ds[split_name].to_pandas()

            # Salva cópia local para garantir funcionamento offline futuro
            try:
                os.makedirs(os.path.dirname(path_to_use), exist_ok=True)
                df.to_csv(path_to_use, index=False)
            except Exception:
                pass

            info["source_used"] = f"Hugging Face Hub ({HF_DATASET_NAME})"
            info["success"] = True
            info["rows"] = len(df)
            info["cols"] = len(df.columns)
            info["message"] = f"Dataset carregado diretamente do Hugging Face com sucesso ({info['rows']} registros, {info['cols']} colunas)."
            return df, info
        except Exception as hf_err:
            if source_preference == "huggingface":
                info["source_used"] = "Hugging Face"
                info["success"] = False
                info["error_details"] = str(hf_err)
                info["message"] = f"Erro ao acessar Hugging Face: {hf_err}"
                return None, info
            # Em modo 'auto', registra o erro e tenta fallback local
            info["error_details"] = f"Tentativa remota falhou ({hf_err}). Recorrendo ao cache local."

    # 3. Fallback ou preferência por CSV local
    if os.path.exists(path_to_use):
        try:
            df = pd.read_csv(path_to_use)
            info["source_used"] = f"Cache Local ({os.path.basename(path_to_use)})"
            info["success"] = True
            info["rows"] = len(df)
            info["cols"] = len(df.columns)
            info["message"] = f"Dataset carregado com sucesso do armazenamento local ({info['rows']} registros, {info['cols']} colunas)."
            return df, info
        except Exception as local_err:
            info["source_used"] = "Local"
            info["success"] = False
            info["error_details"] = str(local_err)
            info["message"] = f"Erro ao ler arquivo CSV local: {local_err}"
            return None, info

    # 4. Nenhuma fonte disponível
    info["success"] = False
    info["message"] = "Nenhuma fonte de dados pôde ser acessada. Conecte-se à internet ou forneça o arquivo CSV local."
    return None, info


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa detalhadamente a estrutura do dataset e gera relatório estatístico sobre colunas, tipos e valores ausentes.
    """
    if df is None or df.empty:
        return {}

    total_rows = len(df)
    total_cols = len(df.columns)
    duplicated_rows = int(df.duplicated().sum())

    # Análise de colunas
    col_analysis = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct_missing = round((n_missing / total_rows) * 100, 2)
        n_unique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)

        # Amostra de valores
        sample_vals = df[col].dropna().head(3).tolist()
        sample_repr = ", ".join(map(str, sample_vals)) if sample_vals else "Vazio"

        col_analysis.append({
            "coluna": col,
            "tipo": dtype_str,
            "valores_ausentes": n_missing,
            "pct_ausentes": pct_missing,
            "valores_unicos": n_unique,
            "exemplo": sample_repr
        })

    col_df = pd.DataFrame(col_analysis)

    # Análise específica do target
    target_summary = {}
    if "total_fatalities" in df.columns:
        valid_target = df["total_fatalities"].dropna()
        target_summary = {
            "total_validos": int(len(valid_target)),
            "nulos": int(df["total_fatalities"].isna().sum()),
            "classe_0_nao_fatais": int((valid_target == 0).sum()),
            "classe_1_fatais": int((valid_target > 0).sum()),
            "pct_fatais": round(float((valid_target > 0).mean() * 100), 2),
            "max_fatalidades": float(valid_target.max()) if len(valid_target) > 0 else 0
        }

    return {
        "total_registros": total_rows,
        "total_colunas": total_cols,
        "duplicatas_completas": duplicated_rows,
        "memoria_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "colunas_detalhes": col_df,
        "target_summary": target_summary
    }
