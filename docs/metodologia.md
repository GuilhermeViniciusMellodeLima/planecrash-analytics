# Metodologia Técnica — PLANECRASH ANALYTICS

**Projeto:** Sistema Inteligente de Classificação da Gravidade de Acidentes Aeronáuticos  
**Autores:**  
- **Guilherme Vinicius Mello de Lima** — Prontuário: `CG3028461`  
- **Luiz Gustavo Alves dos Santos** — Prontuário: `CG302900X`  
**Dataset Base:** `MauroPello/planecrashdata` (Hugging Face Datasets)

---

## 1. Visão Geral e Objetivos do Sistema

O objetivo primordial deste projeto acadêmico consiste em analisar registros históricos mundiais de acidentes aeronáuticos e classificar a gravidade de cada ocorrência em duas classes mutuamente exclusivas:
- **Classe 0:** Acidente sem vítimas fatais (*Non-fatal crash*).
- **Classe 1:** Acidente com vítimas fatais (*Fatal crash*).

O sistema busca correlacionar fatores operacionais pré-impacto (como modelo da aeronave, fase de voo, tipo de operação, condições de solo e contingente a bordo) com a probabilidade estatística de sobrevivência ou ocorrência de mortes.

> [!NOTE]
> O sistema constitui uma ferramenta estritamente acadêmica de ciência de dados e pesquisa exploratória. Não possui certificação operacional por agências reguladoras (como ANAC, FAA, EASA ou ICAO) e não se destina ao prognóstico de segurança de voos futuros.

---

## 2. Origem e Estrutura dos Dados

Os dados foram obtidos diretamente do repositório oficial no Hugging Face:  
🔗 **Fonte:** [https://huggingface.co/datasets/MauroPello/planecrashdata](https://huggingface.co/datasets/MauroPello/planecrashdata)

### 2.1. Métricas da Base Bruta
- **Total de registros catalogados:** 27.598 ocorrências.
- **Total de atributos/colunas:** 28 colunas originais.
- **Amplitude temporal:** Desde o início do século XX até a aviação contemporânea.

### 2.2. Dicionário Original e Classificação no Pipeline

| Coluna | Descrição | Tratamento e Papel no Sistema |
| :--- | :--- | :--- |
| `id` | Identificador único numérico | Metadado — Excluído da modelagem |
| `aircraft` | Modelo ou tipo da aeronave | **Feature Categórica** (One-Hot Encoded) |
| `operator` | Companhia aérea ou operador | Excluída (alta dispersão / ruído textual) |
| `registration` | Matrícula de registro do avião | Metadado — Excluído da modelagem |
| `flight_phase` | Fase do voo na ocorrência | **Feature Categórica** (One-Hot Encoded) |
| `flight_type` | Categoria operacional do voo | **Feature Categórica** (One-Hot Encoded) |
| `survivors` | Contagem de sobreviventes | **DATA LEAKAGE — Estritamente Excluída** |
| `crash_site` | Terreno onde ocorreu a queda | **Feature Categórica** (One-Hot Encoded) |
| `schedule` | Rota / Programação de voo | Metadado — Excluído da modelagem |
| `msn` | Número de série do fabricante | Metadado — Excluído da modelagem |
| `yom` | Ano de fabricação do avião | **Feature Numérica** (Tratada e Padronizada) |
| `crash_location` | Localização descritiva livre | Metadado — Excluído da modelagem |
| `country` | País da ocorrência | Utilizado para análise exploratória |
| `region` | Região ou continente do voo | **Feature Categórica** (One-Hot Encoded) |
| `crew_on_board` | Quantidade de tripulantes a bordo | **Feature Numérica** (Tratada e Padronizada) |
| `crew_fatalities` | Fatalidades entre tripulantes | **DATA LEAKAGE — Estritamente Excluída** |
| `passengers_on_board`| Quantidade de passageiros | **Feature Numérica** (Tratada e Padronizada) |
| `passengers_fatalities`| Fatalidades entre passageiros | **DATA LEAKAGE — Estritamente Excluída** |
| `other_fatalities` | Fatalidades de terceiros em terra | **DATA LEAKAGE — Estritamente Excluída** |
| `total_fatalities` | Total de fatalidades no acidente | **VARIÁVEL-ALVO (TARGET BINÁRIO)** |
| `circumstances` | Narrativa detalhada pós-acidente | **DATA LEAKAGE — Estritamente Excluída** |
| `crash_cause` | Causa primária apurada em laudo | **DATA LEAKAGE — Estritamente Excluída** |
| `timestamp` | Data e hora em formato ISO-8601 | Utilizado em análises de séries temporais |
| `latitude` / `longitude` | Coordenadas decimais | Metadados geográficos brutos |
| `image_url` | Imagem da aeronave/ocorrência | Metadado — Excluído da modelagem |
| `hex_code` | Código ICAO 24-Bit Mode S | Metadado — Excluído da modelagem |
| `external_id` | ID externo de bases associadas | Metadado — Excluído da modelagem |

---

## 3. Prevenção Rigorosa de Data Leakage (Vazamento de Dados)

Um dos erros metodológicos mais graves na literatura de Machine Learning é permitir que o classificador tenha acesso a variáveis que só existem ou só podem ser computadas **após** o desfecho do evento (vazamento temporal ou de rótulo).

No PLANECRASH ANALYTICS, foram instituídas as seguintes barreiras:
1. **Bloqueio Total de Contadores de Vítimas e Sobreviventes:**
   As colunas `crew_fatalities`, `passengers_fatalities`, `other_fatalities` e `survivors` foram estritamente bloqueadas. Se utilizadas, causariam acurácia artificialmente perfeita de 100%, desprovida de qualquer valor analítico real.
2. **Eliminação de Narrativas Periciais Pós-Fato:**
   As colunas `circumstances` e `crash_cause` são relatórios técnicos consolidados por comissões de investigação (como NTSB e CENIPA) meses ou anos após o acidente. Conteriam termos como *"destruição total", "impacto fatal", "sem sobreviventes"*, agindo como vazamento indireto.
3. **Isolamento de Ajuste dos Transformadores:**
   Todo o cálculo de médias, medianas, desvios-padrão (`StandardScaler`) e mapeamento de categorias raras (`OneHotEncoder`) foi ajustado (`fit`) **estritamente sobre o conjunto de treino de 80%**, sendo apenas aplicado (`transform`) nos dados de teste de 20%.

---

## 4. Engenharia de Recursos e Pré-processamento

### 4.1. Construção da Variável-Alvo (Target)
Registros com valor nulo em `total_fatalities` (26 registros) foram descartados. A coluna foi transformada na variável binária `target`:
$$\text{target} = \begin{cases} 0, & \text{se } \text{total\_fatalities} = 0 \text{ (Sem vítimas fatais)} \\ 1, & \text{se } \text{total\_fatalities} > 0 \text{ (Com vítimas fatais)} \end{cases}$$

Distribuição real da base:
- **Classe 0 (Não Fatais):** 11.695 registros (42,42%)
- **Classe 1 (Fatais):** 15.877 registros (57,58%)
- **Total de registros válidos:** 27.572 registros

### 4.2. Saneamento de Inconsistências Históricas
- **Ano de Fabricação (`yom`):** Identificados registros com valores anômalos como `yom = 0` ou `yom = 19567` (erros de digitação em arquivos legados). Valores fora do intervalo realista $[1900, 2026]$ foram convertidos para `NaN` e imputados pela mediana amostral.
- **Tripulação (`crew_on_board`):** Detectado registro histórico com valor 1924 (confusão com ano). Contagens acima de 50 tripulantes foram saneadas.
- **Passageiros (`passengers_on_board`):** O maior avião comercial do mundo (Airbus A380) transporta até 853 passageiros. Valores superiores foram filtrados.
- **Atributo Derivado (`total_on_board`):** Criado pela soma $\text{crew\_on\_board} + \text{passengers\_on\_board}$, representando a carga total de ocupantes.

### 4.3. Pipeline Scikit-Learn
- **Sub-pipeline Numérico:** `SimpleImputer(strategy='median')` seguido de `StandardScaler()`.
- **Sub-pipeline Categórico:** `SimpleImputer(strategy='constant', fill_value='Desconhecido')` seguido de `OneHotEncoder(handle_unknown='ignore', min_frequency=15, sparse_output=False)`.

---

## 5. Modelos de Machine Learning Implementados

Conforme estabelecido no **Entregável 2**, foram desenvolvidos três classificadores supervisionados:

### 5.1. Regressão Logística (*Logistic Regression*)
- **Papel:** Modelo de referência (*Baseline* linear probabilístico).
- **Configuração:** `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`, `random_state=42`.
- **Propósito:** Estimar as probabilidades marginais logarítmicas de ocorrência fatal sob uma superfície linear de decisão.

### 5.2. Árvore de Decisão (*Decision Tree Classifier*)
- **Papel:** Modelo não linear baseado em regras explícitas de particionamento.
- **Configuração:** `max_depth=10`, `min_samples_leaf=10`, `min_samples_split=20`, `class_weight='balanced'`, `random_state=42`.
- **Propósito:** Identificar caminhos de decisão interpretáveis relacionando fase de voo, tipo de aeronave e terreno.

### 5.3. Random Forest (*Floresta Aleatória*)
- **Papel:** Modelo *Ensemble* robusto via agregação por bagging de múltiplas árvores não correlacionadas.
- **Configuração:** `n_estimators=100`, `max_depth=15`, `min_samples_leaf=5`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.
- **Propósito:** Minimizar a variância, mitigar o risco de sobreajuste e maximizar o Recall na identificação de eventos críticos.

---

## 6. Estratégia de Particionamento e Métricas de Desempenho

### 6.1. Divisão Amostral Estratificada
- **Conjunto de Treinamento:** 80% (22.057 amostras)
- **Conjunto de Teste:** 20% (5.515 amostras)
- **Estratificação:** `stratify=y`, garantindo idêntica proporção das duas classes em ambos os subconjuntos.

### 6.2. Métricas Computadas
1. **Acurácia (Accuracy):** Taxa global de acertos $\frac{VP + VN}{Total}$.
2. **Precisão (Precision):** Confiabilidade das predições positivas $\frac{VP}{VP + FP}$.
3. **Recall (Revocação / Sensibilidade):** Capacidade de detectar ocorrências com vítimas $\frac{VP}{VP + FN}$.
4. **F1-Score:** Média harmônica entre Precisão e Recall $2 \times \frac{P \times R}{P + R}$.
5. **ROC-AUC:** Área sob a curva *Receiver Operating Characteristic*.
6. **Matriz de Confusão:** Contabilização explícita de $VN$, $FP$, $FN$ e $VP$.

> [!IMPORTANT]
> **Priorização do Recall:**  
> Na segurança do transporte aéreo, o custo humano e técnico de um **Falso Negativo (FN)** — prever que um acidente não causará vítimas quando ele de fato resulta em fatalidades — é substancialmente mais grave do que um Falso Positivo (alerta conservador). Por esse motivo, a métrica de Recall recebeu a maior ponderação na seleção do melhor algoritmo.
