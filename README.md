# ✈️ PLANECRASH ANALYTICS
### Sistema Inteligente de Classificação da Gravidade de Acidentes Aeronáuticos

> **Projeto Acadêmico de Ciência de Dados, Machine Learning e Large Language Models (LLM)**  
> Desenvolvido em conformidade integral com os requisitos dos **Entregáveis 1 e 2**.

---

## 👥 Integrantes da Equipe

- **Guilherme Vinicius Mello de Lima** — Prontuário: `CG3028461`
- **Luiz Gustavo Alves dos Santos** — Prontuário: `CG302900X`

---

## 📌 Visão Geral do Projeto

O **PLANECRASH ANALYTICS** é uma plataforma acadêmica completa projetada para analisar registros históricos mundiais de acidentes da aviação e classificar a gravidade de cada ocorrência em duas classes fundamentais:
- **Classe 0:** Acidente sem vítimas fatais (*Non-fatal crash* — `total_fatalities = 0`)
- **Classe 1:** Acidente com vítimas fatais (*Fatal crash* — `total_fatalities > 0`)

O sistema correlaciona características operacionais pré-impacto (como fase do voo, modelo da aeronave, tipo de voo, contingente de pessoas a bordo e terreno de queda) com o risco de fatalidades, utilizando três algoritmos supervisionados de Machine Learning e recursos de explicabilidade via Large Language Models (LLM).

---

## 🛡️ Prevenção Rigorosa de Data Leakage (Vazamento de Dados)

Para assegurar validade preditiva real e impedir acurácias artificiais, o sistema **bloqueia estritamente** do conjunto de treinamento qualquer atributo posterior ao acidente ou que dependa diretamente do desfecho:
- ❌ `total_fatalities` (usada unicamente para formular a classe-alvo binária)
- ❌ `crew_fatalities`, `passengers_fatalities`, `other_fatalities` (revelam mortes diretamente)
- ❌ `survivors` (revela sobrevivência pós-fato)
- ❌ `circumstances` e `crash_cause` (narrativas e laudos periciais emitidos após a conclusão das investigações)

Todo o pré-processamento (imputação e codificação *One-Hot*) é calibrado **exclusivamente sobre o conjunto de treino (80%)**, sem contaminação do conjunto de teste (20%).

---

## 🔬 Modelos Implementados e Métricas Reais no Conjunto de Teste (20%)

Avaliação realizada sobre **5.515 instâncias inéditas** com separação estratificada (`stratify=y`, `random_state=42`):

| Modelo | Papel Metodológico | Acurácia | Precisão | Recall (Fatais) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** | *Baseline Linear* | 70,10% | 77,30% | 68,07% | 72,39% | 78,88% |
| **Árvore de Decisão** | *Interpretabilidade* | 71,71% | **77,35%** | 71,95% | 74,55% | 80,94% |
| **Random Forest** | *Ensemble Robusto* | **72,57%** | 76,85% | **74,94%** | **75,88%** | **81,32%** |

> 🏆 **Modelo Vencedor: Random Forest.**  
> Em segurança aeronáutica, o custo de um **Falso Negativo (FN)** — classificar um acidente fatal como sem vítimas — é crítico. O Random Forest alcançou o maior **Recall (74,94%)**, minimizando ocorrências de Falsos Negativos e mantendo o maior **F1-Score (75,88%)** e **ROC-AUC (81,32%)**.

---

## 📂 Estrutura e Organização do Repositório

```text
planecrash-ml/
├── app.py                      # Aplicação Principal (Dashboard e Visão Geral)
├── requirements.txt            # Dependências do ecossistema Python
├── README.md                   # Documentação completa e guia de execução
├── .gitignore                  # Arquivos ignorados pelo controle de versão
├── .env.example                # Modelo de configuração para chaves de LLM
├── .streamlit/
│   └── config.toml             # Configuração do tema escuro e servidor
├── data/
│   ├── planecrash_raw.csv      # Cache local da base de dados histórica
│   └── .gitkeep
├── models/
│   ├── preprocessor.joblib     # Pipeline Scikit-Learn de transformação salvo
│   ├── regressao_logistica.joblib
│   ├── arvore_de_decisao.joblib
│   ├── random_forest.joblib
│   ├── metrics.json            # Métricas e matrizes reais serializadas
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Carregamento do HF e cache local com fallback
│   ├── preprocessing.py        # Limpeza, target binário, pipelines e anti-leakage
│   ├── train.py                # Treinamento dos 3 modelos e persistência Joblib
│   ├── evaluation.py           # Cálculo das métricas oficiais e diagnóstico
│   ├── prediction.py           # Módulo de inferência e validação de cenários
│   ├── visualizations.py       # Gráficos interativos em tema escuro com Plotly
│   ├── llm_service.py          # Integração LLM (OpenAI/Groq/Ollama) e síntese local
│   └── ui_components.py        # Design system, CSS e branding
├── pages/
│   ├── 1_Dataset.py            # Dicionário, integridade, nulos e metadados
│   ├── 2_Analise_Exploratoria.py # Gráficos exploratórios com filtros dinâmicos
│   ├── 3_Treinamento.py        # Treinamento interativo dos 3 modelos
│   ├── 4_Comparacao_Modelos.py # Tabela oficial, matrizes de confusão e curvas ROC
│   ├── 5_Predicao.py           # Simulador preditivo de acidentes em tempo real
│   ├── 6_Analise_IA.py         # Interpretação por IA e geração de texto para artigo
│   └── 7_Sobre_o_Projeto.py    # Autores, alinhamento aos entregáveis e ética
├── tests/
│   ├── test_data_loader.py     # Testes unitários do carregamento
│   ├── test_preprocessing.py   # Testes de target, limpeza e anti-leakage
│   └── test_models.py          # Testes de modelos, métricas e inferência
└── docs/
    ├── metodologia.md          # Especificação técnica e pipeline detalhado
    └── artigo_cientifico_base.md # Estrutura inicial completa do artigo científico
```

---

## ⚙️ Instalação e Execução

### 1. Pré-requisitos
- Python 3.11 ou superior instalado na máquina.
- Conexão com a internet para download inicial (ou utilização do cache local em `data/`).

### 2. Clonagem e Instalação de Dependências
```bash
# Clone o repositório
git clone <URL_DO_REPOSITORIO>
cd planecrash-ml

# Instale os pacotes necessários
pip install -r requirements.txt
```

### 3. Configuração da Chave do LLM (Opcional)
O sistema funciona **100% de forma autônoma mesmo sem chave de API**, gerando sínteses analíticas automáticas a partir das métricas calculadas.  
Para habilitar respostas dinâmicas em linguagem natural via IA:
```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo .env e adicione sua chave:
# OPENAI_API_KEY=sk-...
```

### 4. Executando os Testes Automatizados
```bash
python -m pytest tests -v
```

### 5. Iniciando a Aplicação Web
```bash
streamlit run app.py
```
Acesse no navegador: `http://localhost:8501`.

---

## 📊 Fonte de Dados Oficial

- **Dataset:** *PlaneCrashData Dataset*
- **Autor/Hospedagem:** Mauro Pello via Hugging Face Hub
- **Link Oficial:** [https://huggingface.co/datasets/MauroPello/planecrashdata](https://huggingface.co/datasets/MauroPello/planecrashdata)

---

## ⚠️ Declaração de Limitações Metodológicas e Ética

Este sistema é uma **ferramenta acadêmica de análise histórica de dados** desenvolvida para fins de estudo e pesquisa universitária. **NÃO constitui um sistema de segurança de voo, não substitui laudos periciais de órgãos de investigação da aviação civil (CENIPA, NTSB, ICAO) e não deve ser utilizado como base para navegação aérea ou despacho operacional real.**
