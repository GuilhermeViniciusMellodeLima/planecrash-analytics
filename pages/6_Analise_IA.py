"""
Página: Análise com Inteligência Artificial (LLM) - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st
import pandas as pd

from src.ui_components import inject_custom_css, render_header, render_sidebar_info, render_metric_card
from src.train import are_models_trained, load_saved_artifacts
from src.llm_service import query_llm, get_llm_config, generate_paper_summary

st.set_page_config(
    page_title="PLANECRASH ANALYTICS — Análise com IA",
    page_icon="💡",
    layout="wide"
)

inject_custom_css()
render_sidebar_info()

render_header(
    title="Análise com Inteligência Artificial",
    subtitle="Interpretação Semântica dos Resultados de Machine Learning, Matrizes de Confusão e Geração de Resumo para Artigo Científico"
)

# Verifica se os modelos foram treinados para termos métricas reais
if not are_models_trained():
    st.warning("⚠️ Os modelos de Machine Learning ainda não foram treinados. Treine os modelos para que a IA possa analisar métricas reais.")
    if st.button("Ir para Treinamento dos Modelos"):
        st.switch_page("pages/3_Treinamento.py")
    st.stop()

artifacts = load_saved_artifacts()
if not artifacts:
    st.error("Erro ao carregar os artefatos salvos dos modelos.")
    st.stop()

llm_cfg = get_llm_config()

# Banner de status do LLM
if llm_cfg["is_configured"]:
    st.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 0.9rem 1.4rem; margin-bottom: 1.5rem;">
        <span style="color: #34D399; font-weight: 700;">🟢 Provedor LLM Conectado:</span>
        <span style="color: #E2E8F0; margin-left: 6px;">Modelo: <code>{llm_cfg['model']}</code> | Endpoint: <code>{llm_cfg['base_url']}</code></span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 0.9rem 1.4rem; margin-bottom: 1.5rem;">
        <span style="color: #FBBF24; font-weight: 700;">🟡 Modo Síntese Local Ativo:</span>
        <span style="color: #E2E8F0; margin-left: 6px;">
            A chave de API (<code>OPENAI_API_KEY</code>) não foi configurada no arquivo <code>.env</code>.
            O sistema apresentará sínteses técnicas detalhadas baseadas nas <strong>métricas reais calculadas</strong>.
            Para habilitar consultas dinâmicas com IA generativa, copie o arquivo <code>.env.example</code> para <code>.env</code> e insira sua chave.
        </span>
    </div>
    """, unsafe_allow_html=True)

# Abas temáticas de análise com IA
tab_comp, tab_cm, tab_imp, tab_paper, tab_chat = st.tabs([
    "📊 Interpretação dos 3 Modelos",
    "🔲 Análise da Matriz de Confusão & Recall",
    "⭐ Análise das Variáveis Críticas",
    "📝 Resumo para Artigo Científico",
    "💬 Perguntas Livres sobre os Resultados"
])

with tab_comp:
    st.markdown("#### Explicabilidade Comparativa dos Algoritmos")
    st.caption("A IA analisa as discrepâncias de Acurácia, Precisão, Recall e F1-Score entre Regressão Logística, Árvore de Decisão e Random Forest.")

    if st.button("Gerar Análise Comparativa dos Modelos", key="btn_ai_comp"):
        with st.spinner("Processando diagnóstico com dados reais..."):
            prompt = (
                "Analise detalhadamente o desempenho comparativo dos três modelos (Regressão Logística, Árvore de Decisão e Random Forest) "
                "no conjunto de teste. Explique a relevância do baseline de Regressão Logística e por que o ensemble Random Forest se destacou."
            )
            res = query_llm(prompt=prompt, context_data=artifacts)
            st.markdown(res["response"])
    else:
        st.info("Clique no botão acima para sintetizar a interpretação técnica dos três modelos.")

with tab_cm:
    st.markdown("#### Interpretação das Matrizes de Confusão e Trade-off de Erro")
    st.caption("Entenda o impacto humano e operacional de Falsos Negativos vs Falsos Positivos na aviação.")

    if st.button("Analisar Matrizes de Confusão e Falsos Negativos", key="btn_ai_cm"):
        with st.spinner("Calculando impacto dos quadrantes de erro..."):
            prompt = (
                "Interprete as matrizes de confusão dos três modelos. Explique didaticamente a diferença entre "
                "Falsos Positivos (alarmes falsos) e Falsos Negativos (acidentes com vítimas não detectados), "
                "demonstrando por que a maximização do Recall é o principal critério de segurança adotado no projeto."
            )
            res = query_llm(prompt=prompt, context_data=artifacts)
            st.markdown(res["response"])
    else:
        st.info("Clique no botão acima para analisar a distribuição dos erros de classificação.")

with tab_imp:
    st.markdown("#### Interpretação das Variáveis Mais Importantes")
    st.caption("A IA explica como a física e a operação de voo refletem no ranking de variáveis do Random Forest.")

    if st.button("Interpretar Variáveis Mais Importantes", key="btn_ai_imp"):
        with st.spinner("Avaliando significância das variáveis..."):
            prompt = (
                "Interprete o ranking das variáveis de maior importância identificadas pelo modelo Random Forest. "
                "Explique como fatores como 'fase do voo', 'local de queda' e 'número de pessoas a bordo' afetam "
                "a probabilidade estatística de sobrevivência ou fatalidade."
            )
            res = query_llm(prompt=prompt, context_data=artifacts)
            st.markdown(res["response"])
    else:
        st.info("Clique no botão acima para compreender a física e as correlações aprendidas pelo modelo.")

with tab_paper:
    st.markdown("#### Geração de Conteúdo para o Artigo Científico")
    st.markdown("""
    Este módulo redige um texto acadêmico fundamentado para a seção de **Resultados e Discussão** do artigo,
    atribuído aos alunos **Guilherme Vinicius Mello de Lima (CG3028461)** e **Luiz Gustavo Alves dos Santos (CG302900X)**.
    """)

    if st.button("📄 Gerar Seção de Resultados para Artigo Científico", type="primary", key="btn_ai_paper"):
        with st.spinner("Redigindo texto acadêmico fundamentado..."):
            summary_text = generate_paper_summary(artifacts)
            st.markdown(summary_text)

            st.download_button(
                label="⬇️ Baixar Texto em Markdown (.md)",
                data=summary_text,
                file_name="resultados_discussao_artigo.md",
                mime="text/markdown"
            )

with tab_chat:
    st.markdown("#### Chat Interativo com IA sobre o Projeto")
    st.markdown("Faça perguntas sobre as métricas reais, a base de dados histórica ou as decisões metodológicas do projeto:")

    user_query = st.text_input(
        "Sua dúvida ou solicitação:",
        placeholder="Ex: Por que a árvore de decisão teve recall superior à regressão logística?"
    )

    if st.button("Enviar Pergunta", key="btn_ai_ask") and user_query:
        with st.spinner("Analisando com base nos dados do sistema..."):
            res = query_llm(prompt=user_query, context_data=artifacts)
            st.markdown("##### Resposta do Especialista:")
            st.markdown(res["response"])
