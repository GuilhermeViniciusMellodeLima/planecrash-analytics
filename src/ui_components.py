"""
Módulo de Componentes de Interface e Estilo - PLANECRASH ANALYTICS
Padroniza o design system em tema escuro (azul-marinho, grafite e ciano),
cards modernos, cabeçalhos estilizados e identificação acadêmica dos integrantes.

Autores:
  - Guilherme Vinicius Mello de Lima (CG3028461)
  - Luiz Gustavo Alves dos Santos (CG302900X)
"""

import streamlit as st


def inject_custom_css():
    """Injeta estilos CSS personalizados para um visual sofisticado e responsivo."""
    st.markdown("""
    <style>
        /* Tipografia e Fundo Geral */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Container Principal */
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        /* Topbar / Header Banner */
        .analytics-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2847 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.8rem 2.2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(0, 210, 255, 0.08);
            position: relative;
            overflow: hidden;
        }
        
        .analytics-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 5px;
            height: 100%;
            background: linear-gradient(180deg, #00D2FF 0%, #3B82F6 100%);
        }

        .analytics-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #F8FAFC;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .analytics-title span {
            background: linear-gradient(90deg, #00D2FF 0%, #60A5FA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .analytics-subtitle {
            font-size: 1.02rem;
            font-weight: 400;
            color: #94A3B8;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
        }

        .analytics-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 0.8rem;
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .badge-cyan {
            background: rgba(0, 210, 255, 0.12);
            color: #00D2FF;
            border: 1px solid rgba(0, 210, 255, 0.3);
        }

        .badge-slate {
            background: rgba(148, 163, 184, 0.12);
            color: #CBD5E1;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        .badge-success {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-danger {
            background: rgba(244, 63, 94, 0.15);
            color: #FB7185;
            border: 1px solid rgba(244, 63, 94, 0.3);
        }

        /* Metric Cards */
        .metric-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease-in-out;
            height: 100%;
        }

        .metric-card:hover {
            border-color: #00D2FF;
            transform: translateY(-2px);
            box-shadow: 0 8px 16px -2px rgba(0, 210, 255, 0.15);
        }

        .metric-label {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            margin-bottom: 0.3rem;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.2;
        }

        .metric-hint {
            font-size: 0.78rem;
            color: #64748B;
            margin-top: 0.35rem;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0A0F1D !important;
            border-right: 1px solid #1E293B;
        }

        .sidebar-authors-box {
            background: #131D31;
            border: 1px solid #233554;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1.5rem;
            font-size: 0.85rem;
        }

        .sidebar-author-name {
            font-weight: 600;
            color: #E2E8F0;
        }

        .sidebar-author-id {
            color: #00D2FF;
            font-family: monospace;
            font-size: 0.82rem;
        }

        /* Botões Estilizados */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.25rem;
            transition: all 0.2s ease;
        }

        .stButton>button:hover {
            border-color: #00D2FF;
            box-shadow: 0 0 12px rgba(0, 210, 255, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str = "PLANECRASH ANALYTICS", subtitle: str = "Sistema Inteligente de Análise e Classificação de Acidentes Aeronáuticos"):
    """Renderiza o banner superior padrão com branding do projeto."""
    st.markdown(f"""
    <div class="analytics-header">
        <h1 class="analytics-title">
            ✈️ <span>{title}</span>
        </h1>
        <div class="analytics-subtitle">{subtitle}</div>
        <div class="analytics-badge-row">
            <span class="badge-pill badge-cyan">Machine Learning Supervisionado</span>
            <span class="badge-pill badge-slate">Dataset: MauroPello/planecrashdata</span>
            <span class="badge-pill badge-slate">Target: total_fatalities (Binário)</span>
            <span class="badge-pill badge-success">Ambiente Acadêmico</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_info():
    """Renderiza as informações institucionais dos alunos e projeto na barra lateral."""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 0.8rem 0 1.2rem 0;">
        <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">
            ✈️ <span style="color: #00D2FF;">PLANECRASH</span> ANALYTICS
        </div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">
            Classificação Inteligente de Acidentes
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div class="sidebar-authors-box">
        <div style="font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; margin-bottom: 6px;">
            👥 Integrantes da Equipe
        </div>
        <div style="margin-bottom: 8px;">
            <div class="sidebar-author-name">Guilherme Vinicius Mello de Lima</div>
            <div class="sidebar-author-id">Prontuário: CG3028461</div>
        </div>
        <div>
            <div class="sidebar-author-name">Luiz Gustavo Alves dos Santos</div>
            <div class="sidebar-author-id">Prontuário: CG302900X</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")


def render_metric_card(label: str, value: str, hint: str = "", border_color: str = "#334155"):
    """Renderiza um card moderno para métricas numéricas."""
    st.markdown(f"""
    <div class="metric-card" style="border-color: {border_color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-hint">{hint}</div>' if hint else ''}
    </div>
    """, unsafe_allow_html=True)
