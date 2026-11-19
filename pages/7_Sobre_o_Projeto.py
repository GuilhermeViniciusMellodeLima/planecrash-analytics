"""
Página: Sobre o Projeto e Documentação Institucional - PLANECRASH ANALYTICS
Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st

from src.ui_components import inject_custom_css, render_header, render_sidebar_info

st.set_page_config(
    page_title="PLANECRASH ANALYTICS — Sobre o Projeto",
    page_icon="ℹ️",
    layout="wide"
)

inject_custom_css()
render_sidebar_info()

render_header(
    title="Sobre o Projeto PLANECRASH ANALYTICS",
    subtitle="Identificação dos Integrantes, Conformidade com Entregáveis 1 e 2, Arquitetura e Diretrizes Éticas"
)

# Identificação dos Alunos
st.markdown("### 👥 Identificação dos Integrantes da Equipe")

col_a1, col_a2 = st.columns(2)

with col_a1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Aluno 1</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">Guilherme Vinicius Mello de Lima</div>
        <div style="font-size: 1rem; color: #00D2FF; font-family: monospace; margin-top: 2px;">Prontuário: CG3028461</div>
        <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 8px;">Desenvolvimento Full Stack, Engenharia de ML e Arquitetura de Software.</div>
    </div>
    """, unsafe_allow_html=True)

with col_a2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Aluno 2</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">Luiz Gustavo Alves dos Santos</div>
        <div style="font-size: 1rem; color: #00D2FF; font-family: monospace; margin-top: 2px;">Prontuário: CG302900X</div>
        <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 8px;">Ciência de Dados, Análise Exploratória e Modelagem Preditiva.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Conformidade com os Entregáveis
st.markdown("### 📑 Alinhamento com os Entregáveis Oficiais da Disciplina")

c_e1, c_e2 = st.columns(2)

with c_e1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-weight: 700; color: #00D2FF; font-size: 1.1rem; margin-bottom: 8px;">Entregável 1: Proposta e Dataset</div>
        <ul style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.6; padding-left: 20px; margin: 0;">
            <li><strong>Dataset Oficial:</strong> <code>MauroPello/planecrashdata</code> (Hugging Face).</li>
            <li><strong>Volume:</strong> 27.598 registros históricos e 28 colunas documentadas.</li>
            <li><strong>Dicionário Completo:</strong> Mapeamento integral de atributos técnicos e contextuais.</li>
            <li><strong>Variável-Alvo (Target):</strong> Derivada da coluna <code>total_fatalities</code>.</li>
            <li><strong>Binário:</strong> Classe 0 (0 fatalidades) e Classe 1 (> 0 fatalidades).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with c_e2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-weight: 700; color: #00D2FF; font-size: 1.1rem; margin-bottom: 8px;">Entregável 2: Modelos e Métricas</div>
        <ul style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.6; padding-left: 20px; margin: 0;">
            <li><strong>Modelos Obrigatórios:</strong> Regressão Logística (Baseline), Árvore de Decisão e Random Forest.</li>
            <li><strong>Divisão Amostral:</strong> 80% Treinamento / 20% Teste com estratificação por classe (<code>stratify=y</code>).</li>
            <li><strong>Métricas de Avaliação:</strong> Acurácia, Precisão, Recall, F1-Score, ROC-AUC e Matriz de Confusão.</li>
            <li><strong>Foco em Segurança:</strong> Maximização do Recall para minimizar Falsos Negativos.</li>
            <li><strong>Prevenção de Data Leakage:</strong> Bloqueio total de variáveis dependentes de sobreviventes e laudos periciais pós-fato.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tecnologias e Arquitetura
st.markdown("### 🛠️ Tecnologias Utilizadas")

t_col1, t_col2, t_col3, t_col4 = st.columns(4)

with t_col1:
    st.markdown("""
    **Backend e ML**
    - Python 3.11+ / 3.14
    - Scikit-learn (Pipelines, Modelos)
    - Pandas & NumPy
    - Joblib (Serialização)
    """)

with t_col2:
    st.markdown("""
    **Interface do Usuário**
    - Streamlit
    - CSS3 Modular & Dark Theme
    - Componentes Dinâmicos
    - Multipage Navigation
    """)

with t_col3:
    st.markdown("""
    **Visualização de Dados**
    - Plotly Graph Objects
    - Plotly Express
    - Heatmaps de Confusão
    - Curvas ROC Integradas
    """)

with t_col4:
    st.markdown("""
    **IA Generativa (LLM)**
    - API compatível com OpenAI
    - Suporte a Groq, Ollama, etc.
    - Prompts Contextualizados
    - Fallback Heurístico Local
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Aviso de Limitações Metodológicas e Éticas
st.markdown("""
> [!CAUTION]
> ### Declaração de Limitações Metodológicas e Ética em Inteligência Artificial
> 1. **Finalidade Estritamente Acadêmica:** O *PLANECRASH ANALYTICS* é uma ferramenta de estudo e pesquisa aplicada desenvolvida
>    no âmbito universitário para exercitar conceitos de classificação supervisionada, tratamento de dados tabulares e explicabilidade.
> 2. **Não Certificação Operacional:** Este software **NÃO constitui um sistema de segurança de voo, não possui homologação por
>    órgãos reguladores da aviação civil (como ANAC, FAA, EASA ou ICAO)** e **NÃO deve ser utilizado para tomada de decisões operacionais**,
>    planejamento de rotas ou despachos de voo.
> 3. **Natureza Histórica dos Dados:** A base de dados abrange desde o início do século XX até períodos recentes.
>    Mudanças drásticas nas tecnologias de navegação aérea, materiais compostos e protocolos de treinamento (CRM)
>    significam que correlações de aeronaves antigas não refletem diretamente o risco da aviação comercial a jato contemporânea.
""")
