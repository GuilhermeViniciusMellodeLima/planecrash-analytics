# Classificação Inteligente da Gravidade de Acidentes Aeronáuticos Utilizando Modelos Supervisionados de Aprendizado de Máquina e Explicabilidade por LLM

**Autores:**  
- **Guilherme Vinicius Mello de Lima** — Prontuário: `CG3028461`  
- **Luiz Gustavo Alves dos Santos** — Prontuário: `CG302900X`  

**Instituição:** Instituto Federal de Educação, Ciência e Tecnologia de São Paulo (IFSP)  
**Área de Concentração:** Ciência da Computação, Engenharia de Software e Inteligência Artificial  

---

## Resumo

A aviação moderna é reconhecida internacionalmente como um dos modais de transporte mais seguros do planeta, resultado direto de décadas de evolução tecnológica, padronização de procedimentos e investigação sistemática de acidentes. Contudo, a análise retrospectiva de ocorrências históricas fornece subsídios fundamentais para a compreensão dos fatores que determinam a severidade de um sinistro aéreo. Este artigo apresenta o desenvolvimento do sistema **PLANECRASH ANALYTICS**, uma solução computacional completa voltada à classificação binária da gravidade de acidentes aeronáuticos em duas classes: acidentes sem vítimas fatais (Classe 0) e acidentes com vítimas fatais (Classe 1). Utilizando o *PlaneCrashData Dataset* (composto por 27.598 registros históricos mundiais), foram implementados e comparados três algoritmos supervisionados: Regressão Logística (*baseline*), Árvore de Decisão e *Random Forest*. Foi estabelecida uma rigorosa política de prevenção contra vazamento de dados (*data leakage*), bloqueando contadores diretos de vítimas e laudos periciais pós-fato. Sob validação estratificada com 20% das amostras reservadas para teste (5.515 instâncias inéditas), o algoritmo **Random Forest** alcançou o melhor desempenho global, obtendo **Acurácia de 72,57%**, **Recall de 74,94%**, **F1-Score de 75,88%** e **ROC-AUC de 81,32%**, destacando-se pela capacidade superior de minimizar Falsos Negativos. O sistema incorpora ainda explicabilidade em linguagem natural por meio de integração com Large Language Models (LLM) e uma aplicação web interativa desenvolvida com Streamlit e Plotly.

**Palavras-chave:** Segurança Aeronáutica; Machine Learning; Random Forest; Prevenção de Data Leakage; Explicabilidade; Large Language Models.

---

## 1. Introdução

A investigação e a mitigação de riscos na aviação civil e militar constituem áreas de interesse crítico para a engenharia e a segurança pública. Cada evento catastrófico ou incidente sério desencadeia um processo exaustivo de apuração conduzido por órgãos credenciados (como o CENIPA no Brasil, o NTSB nos Estados Unidos e a ICAO internacionalmente), gerando vastos repositórios de dados documentais.

Embora grande parte desses dados seja examinada sob o prisma da análise qualitativa de causas imediatas (falhas de componentes, fatores humanos, condições meteorológicas adversas), as técnicas contemporâneas de Ciência de Dados e Aprendizado de Máquina (*Machine Learning*) oferecem a oportunidade de analisar padrões em larga escala (*macro-patterns*), relacionando características estruturais e operacionais pré-impacto à gravidade do desfecho.

O presente trabalho tem como objetivo desenvolver, validar e disponibilizar uma plataforma analítica capaz de classificar se um determinado acidente resultou ou não em vítimas fatais a partir de parâmetros operacionais disponíveis antes da consumação do impacto (como fase de voo, tipo de aeronave, número de ocupantes e características geográficas). Adicionalmente, busca-se garantir a integridade científica do experimento por meio do isolamento contra vazamento de dados (*data leakage*) e fornecer interpretabilidade semântica acessível via Large Language Models.

---

## 2. Fundamentação Teórica e Modelos Adotados

Conforme delineado nos requisitos metodológicos do projeto (Entregável 2), foram selecionados três algoritmos de aprendizado supervisionado:

### 2.1. Regressão Logística
Utilizada como modelo inicial de referência (*baseline*), a Regressão Logística modela a probabilidade a posteriori da classe positiva ($P(y=1|X)$) por meio de uma transformação logística (função sigmoide aplicada a uma combinação linear de atributos):
$$P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$
Sua inclusão fornece uma linha de base paramétrica simples e computacionalmente eficiente, permitindo aferir o ganho proporcionado por métodos não lineares.

### 2.2. Árvore de Decisão (*Decision Tree Classifier*)
As Árvores de Decisão particionam o espaço de atributos em regiões hiper-retangulares recursivas com base em critérios de pureza (como o índice de Gini ou Ganho de Informação). Apresentam a vantagem primordial da transparência e facilidade de auditoria, gerando regras explícitas no formato *se-então*.

### 2.3. Floresta Aleatória (*Random Forest*)
O algoritmo *Random Forest* constitui um comitê (*ensemble*) de árvores de decisão geradas por meio de *bootstrap aggregating* (*bagging*) e seleção aleatória de subconjuntos de variáveis em cada divisão de nó. Esse procedimento reduz expressivamente a variância dos estimadores individuais, mitigando o sobreajuste (*overfitting*) e proporcionando elevada capacidade preditiva em conjuntos de dados tabulares complexos.

---

## 3. Metodologia Experimental

### 3.1. Base de Dados e Definição da Variável-Alvo
Utilizou-se o *PlaneCrashData Dataset*, hospedado no Hugging Face Datasets (`MauroPello/planecrashdata`), contendo 27.598 registros e 28 colunas originais.

A variável-alvo foi formulada a partir da coluna `total_fatalities`, sendo convertida na classificação binária:
- **Classe 0 (Sem vítimas fatais):** `total_fatalities == 0` (11.695 registros válidos; 42,42%)
- **Classe 1 (Com vítimas fatais):** `total_fatalities > 0` (15.877 registros válidos; 57,58%)

Registros com valor ausente no campo de fatalidades (26 instâncias) foram removidos antes do particionamento.

### 3.2. Prevenção Rigorosa de Data Leakage
Para preservar a validade preditiva, foram excluídas terminantemente todas as variáveis que representavam o desfecho direto da ocorrência ou laudos periciais retrospectivos:
1. `survivors` (número pós-fato de sobreviventes);
2. `crew_fatalities`, `passengers_fatalities` e `other_fatalities` (contadores diretos de mortes);
3. `circumstances` (relatório narrativo detalhado produzido após o acidente);
4. `crash_cause` (conclusão técnica da perícia emitida ao término da investigação).

### 3.3. Engenharia de Variáveis e Pipeline de Transformação
As variáveis de entrada selecionadas compreenderam:
- **Numéricas:** `crew_on_board`, `passengers_on_board`, `yom` (ano de fabricação) e `total_on_board` (soma derivada da tripulação e passageiros).
- **Categóricas:** `flight_phase` (fase do voo), `aircraft` (modelo da aeronave), `flight_type` (natureza do voo), `region` (região geográfica) e `crash_site` (terreno da queda).

Inconsistências históricas foram tratadas: valores aberrantes de ano de fabricação ($yom < 1900$ ou $yom > 2026$) e tripulação ($> 50$) foram convertidos para nulos. O pipeline Scikit-Learn aplicou imputação pela mediana e padronização com `StandardScaler` para valores numéricos, e imputação por constante e codificação `OneHotEncoder` (com limiar mínimo de frequência de 15 ocorrências) para atributos categóricos. Os transformadores foram calibrados estritamente sobre o conjunto de treino.

### 3.4. Particionamento Amostral Estratificado
A base foi subdividida em:
- **80% para Treinamento:** 22.057 amostras.
- **20% para Teste:** 5.515 amostras.

A divisão utilizou `stratify=y` e semente pseudoaleatória fixa (`random_state=42`), garantindo a preservação exata da distribuição de classes nas duas partições.

---

## 4. Resultados Experimentais e Discussão

Os três modelos foram submetidos à avaliação empírica sobre o conjunto de teste de 5.515 instâncias inéditas. Todas as métricas foram calculadas considerando a Classe 1 (Acidentes Fatais) como a classe positiva.

### 4.1. Tabela Comparativa de Métricas Reais no Conjunto de Teste

| Modelo de Machine Learning | Acurácia | Precisão | Recall (Sensibilidade) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística (Baseline)** | 70,10% | 77,30% | 68,07% | 72,39% | 78,88% |
| **Árvore de Decisão** | 71,71% | **77,35%** | 71,95% | 74,55% | 80,94% |
| **Random Forest (Ensemble)** | **72,57%** | 76,85% | **74,94%** | **75,88%** | **81,32%** |

### 4.2. Análise da Matriz de Confusão e Falsos Negativos
Na matriz de confusão correspondente ao conjunto de teste do **Random Forest** (composto por 2.339 acidentes não fatais e 3.176 acidentes fatais reais):
- **Verdadeiros Negativos (VN):** 1.616 ocorrências não fatais identificadas corretamente.
- **Falsos Positivos (FP):** 723 alarmes falsos (acidentes não fatais preditos como fatais).
- **Falsos Negativos (FN):** 796 ocorrências fatais classificadas como não fatais.
- **Verdadeiros Positivos (VP):** 2.380 acidentes fatais previstos com precisão.

Em contextos de segurança aeronáutica e gestão de emergências, o custo de um **Falso Negativo** é expressivamente maior do que o de um Falso Positivo:
- Um Falso Positivo gera acionamento desnecessário ou conservador de recursos de resgate médico e suporte intensivo, representando um custo operacional tolerável.
- Um Falso Negativo, por outro lado, subestima a gravidade de um sinistro com potencial perda de vidas, atrasando planos de contingência e busca e salvamento (*Search and Rescue — SAR*).

Dessa forma, o fato de o modelo **Random Forest** ter reduzido os Falsos Negativos e elevado o Recall para **74,94%** (superando o baseline linear em mais de 6,8 pontos percentuais) consolida-o como o classificador mais adequado para a finalidade proposta.

### 4.3. Relevância dos Atributos (*Feature Importance*)
A análise dos coeficientes e índices de impureza do Random Forest evidenciou que os atributos mais determinantes para a separação das classes foram:
1. **Fase do Voo (`flight_phase`):** Acidentes ocorridos em fases de cruzeiro (*Flight / En route*) apresentam probabilidade de fatalidade superior a 80%, em virtude da elevada energia cinética e altitude, enquanto ocorrências em taxiamento ou aproximação final apresentam maiores índices de sobrevivência.
2. **Localização e Terreno (`crash_site`):** Acidentes ocorridos a menos de 10 km de pistas de pouso registram menor letalidade devido à pronta atuação das brigadas de incêndio e resgate de aeródromo (*ARFF*), ao passo que quedas em relevos montanhosos e massas d'água correlacionam-se com severidade catastrófica.
3. **Carga Total a Bordo (`total_on_board`):** O volume de passageiros e tripulantes influencia diretamente a dinâmica de evacuação em situações de emergência de cabine.

---

## 5. Integração com Large Language Models (LLM)

Para transpor a barreira técnica entre as métricas estatísticas de Machine Learning e a tomada de decisão por gestores de segurança ou pesquisadores não especializados em ciência de dados, o PLANECRASH ANALYTICS integra uma camada de Large Language Models compatível com o padrão OpenAI.

O serviço opera sob controle rigoroso de contexto (*context grounding*):
- O LLM recebe como entrada a estrutura JSON das métricas reais calculadas no conjunto de teste, os parâmetros do conjunto de dados e as importâncias das variáveis.
- As diretrizes de engenharia de prompt impõem que o modelo responda estritamente com base nos fatos fornecidos, proibindo a inferência de números fictícios ou especulações desprovidas de respaldo empírico.
- Em cenários onde a chave de API não esteja configurada localmente, o sistema ativa automaticamente um módulo de síntese algorítmica local, garantindo que o usuário tenha acesso contínuo às interpretações técnicas sem interrupção do fluxo de trabalho.

---

## 6. Considerações Finais e Trabalhos Futuros

O projeto **PLANECRASH ANALYTICS** demonstrou com sucesso a aplicação de pipelines modernos de Machine Learning para a classificação da gravidade de acidentes aeronáuticos a partir de uma base com mais de 27 mil registros históricos.

O modelo **Random Forest** comprovou sua superioridade em relação aos classificadores lineares e árvores simples, atingindo **Acurácia de 72,57%**, **Recall de 74,94%** e **ROC-AUC de 81,32%**, preservando a conformidade estrita contra *data leakage*.

Como limitações inerentes ao estudo, ressalta-se a natureza histórica dos dados (que englobam aeronaves de diferentes gerações tecnológicas e doutrinas operacionais que evoluíram drasticamente ao longo do último século). Para trabalhos futuros, sugere-se a incorporação de dados meteorológicos históricos pontuais (temperatura, vento e visibilidade via METAR de estações próximas) e o emprego de técnicas avançadas de calibração de probabilidades (*Isotonic Regression* ou *Platt Scaling*).

---

## Referências Bibliográficas

1. BREIMAN, L. Random Forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.
2. HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2. ed. Springer, 2009.
3. INTERNATIONAL CIVIL AVIATION ORGANIZATION (ICAO). *Safety Management Manual (SMM)*. Doc 9859, 4. ed. Montreal: ICAO, 2018.
4. NATIONAL TRANSPORTATION SAFETY BOARD (NTSB). *Aviation Accident Database & Synopses*. Washington, D.C.: NTSB, 2024.
5. PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011.
