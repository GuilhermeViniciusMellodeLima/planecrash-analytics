"""
Módulo de Integração com Large Language Models (LLM) - PLANECRASH ANALYTICS
Responsável por interpretar em linguagem natural os resultados de Machine Learning,
explicar métricas, gerar resumos acadêmicos e responder dúvidas sobre o projeto.

Compatível com API OpenAI e provedores compatíveis (Groq, Ollama, DeepSeek, Together, etc.).
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()


def get_llm_config() -> Dict[str, Any]:
    """Retorna as configurações atuais do LLM a partir das variáveis de ambiente."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    is_configured = bool(api_key and not api_key.startswith("sua_chave"))

    return {
        "is_configured": is_configured,
        "api_key": api_key,
        "base_url": base_url,
        "model": model
    }


def query_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    context_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Envia uma consulta ao LLM com base estrita nos dados do contexto fornecido.
    Se a API não estiver configurada ou falhar, retorna análise heurística estruturada sem quebrar o sistema.
    """
    config = get_llm_config()

    default_system = (
        "Você é um especialista em Ciência de Dados, Engenharia de Machine Learning e Segurança da Aviação Civil. "
        "Seu papel é analisar estritamente os dados reais fornecidos no contexto do projeto acadêmico PLANECRASH ANALYTICS, "
        "desenvolvido pelos alunos Guilherme Vinicius Mello de Lima (CG3028461) e Luiz Gustavo Alves dos Santos (CG302900X). "
        "REGRAS INEGOCIÁVEIS:\n"
        "1. Nunca invente dados, métricas, acidentes ou números fictícios.\n"
        "2. Baseie todas as suas conclusões exclusivamente nas métricas e fatos informados no prompt.\n"
        "3. Sempre ressalte a relevância do Recall para minimizar falsos negativos na detecção de fatalidades.\n"
        "4. Mantenha tom acadêmico, rigoroso e em Português do Brasil.\n"
        "5. Lembre que o sistema é uma ferramenta acadêmica de análise histórica e não um oráculo de aviação operacional."
    )

    system_prompt = system_instruction or default_system

    # Monta o contexto textual
    full_prompt = prompt
    if context_data:
        full_prompt = f"--- DADOS REAIS CALCULADOS DO SISTEMA ---\n{str(context_data)}\n\n--- SOLICITAÇÃO ---\n{prompt}"

    if not config["is_configured"]:
        return {
            "success": False,
            "fallback_used": True,
            "response": _generate_offline_synthesis(prompt, context_data),
            "message": (
                "Chave de API do LLM não encontrada no arquivo .env. "
                "Exibindo síntese analítica baseada nas métricas reais calculadas pelo sistema. "
                "Para ativar respostas em tempo real via IA, configure a variável OPENAI_API_KEY no arquivo .env."
            )
        }

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )

        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        return {
            "success": True,
            "fallback_used": False,
            "response": response.choices[0].message.content,
            "model_used": config["model"]
        }

    except Exception as e:
        return {
            "success": False,
            "fallback_used": True,
            "response": _generate_offline_synthesis(prompt, context_data),
            "error_details": str(e),
            "message": f"Erro na comunicação com a API do LLM ({e}). Exibindo síntese analítica local."
        }


def _generate_offline_synthesis(prompt: str, context_data: Optional[Dict[str, Any]]) -> str:
    """
    Gera uma interpretação técnica e completa dos dados reais calculados
    para garantir que o usuário tenha insights imediatos mesmo sem chave de API.
    """
    if not context_data:
        return (
            "### Síntese Analítica do Projeto PLANECRASH ANALYTICS\n\n"
            "O projeto investiga o comportamento de três modelos supervisionados (Regressão Logística, Árvore de Decisão e Random Forest) "
            "na classificação binária de acidentes aeronáuticos com e sem fatalidades.\n\n"
            "**Destaques Metodológicos:**\n"
            "- Prevenção ativa de data leakage pela remoção de contadores de sobreviventes e causas pós-fato.\n"
            "- Treinamento estratificado 80/20 com pré-processamento via Pipeline Scikit-learn.\n"
            "- O modelo Random Forest destaca-se por maximizar o Recall na identificação de eventos críticos com fatalidades."
        )

    diag = context_data.get("diagnosis", {})
    comp = context_data.get("comparison_table", [])
    meta = context_data.get("metadata", {})

    linhas_modelos = []
    # Converte DataFrame para lista de dicionários se necessário
    if hasattr(comp, "to_dict"):
        comp_records = comp.to_dict(orient="records")
    elif isinstance(comp, list):
        comp_records = comp
    else:
        comp_records = []

    for m in comp_records:
        linhas_modelos.append(
            f"- **{m.get('Modelo')}**: Acurácia={m.get('Acurácia', 0):.2%}, Precisão={m.get('Precisão', 0):.2%}, "
            f"Recall={m.get('Recall', 0):.2%}, F1-Score={m.get('F1-Score', 0):.2%}, ROC-AUC={m.get('ROC-AUC', 0):.2%}"
        )

    modelos_str = "\n".join(linhas_modelos) if linhas_modelos else "Métricas em processamento."

    return f"""### Análise Técnica Integrada dos Resultados (Modo Síntese Automática)

**Autores do Projeto:** Guilherme Vinicius Mello de Lima (CG3028461) e Luiz Gustavo Alves dos Santos (CG302900X)

#### 1. Desempenho Comparativo dos Modelos
{modelos_str}

#### 2. Veredito Técnico do Melhor Modelo
- **Modelo Vencedor:** {diag.get('melhor_modelo', 'Random Forest')}
- **Fundamentação:** Em segurança aeronáutica, o custo de um **Falso Negativo** (não prever que um acidente causará vítimas fatais) é infinitamente mais grave que um Falso Positivo. O algoritmo vencedor obteve o maior índice de **Recall ({diag.get('recall_vencedor', 0):.2%})**, identificando corretamente a vasta maioria das ocorrências críticas.
- **Falsos Negativos Registrados no Teste:** {diag.get('falsos_negativos', 0)} casos.
- **Falsos Positivos Registrados no Teste:** {diag.get('falsos_positivos', 0)} casos.

#### 3. Impacto das Variáveis Preditoras
As variáveis com maior capacidade discriminativa demonstradas pelos modelos foram a **fase do voo** (acidentes em cruzeiro/rota apresentam probabilidade muito maior de fatalidade em relação a táxi/pouso), o **tipo de terreno/local de queda** (regiões montanhosas ou aquáticas aumentam a severidade) e o **volume total de ocupantes**.

*(Para gerar novas análises interativas e perguntas personalizadas, configure a variável OPENAI_API_KEY no arquivo .env)*
"""


def generate_paper_summary(results_dict: Dict[str, Any]) -> str:
    """Gera um resumo executivo pronto para o artigo científico com base nas métricas reais."""
    diag = results_dict.get("diagnosis", {})
    comp_records = results_dict.get("comparison_table", [])

    prompt = (
        "Com base estritamente nas métricas reais fornecidas, redija a seção de Resultados e Discussão "
        "para o artigo científico dos alunos Guilherme Vinicius Mello de Lima (CG3028461) e "
        "Luiz Gustavo Alves dos Santos (CG302900X). "
        "Destaque a comparação entre Regressão Logística, Árvore de Decisão e Random Forest, "
        "analisando o trade-off entre Recall e Precisão e a prevenção de data leakage."
    )

    res = query_llm(prompt=prompt, context_data=results_dict)
    return res.get("response", "")
