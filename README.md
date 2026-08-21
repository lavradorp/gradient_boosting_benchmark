# Benchmark de Frameworks de Gradient Boosting

Comparação de desempenho estatístico e computacional entre três frameworks de Gradient Boosting: **XGBoost**, **LightGBM** e **CatBoost**.

Tipo do problema de Machine Learning usado: Regressão.

Métricas estatísticas analisadas: MAE, RMSE, R².
Métricas computacionais analisadas: Tempo de treino, tempo de inferência e uso de RAM.

Todos os modelos foram treinados e avaliados sob as mesmas condições: mesma divisão de dados, mesmo pipeline de pré-processamento e mesma quantidade de execuções, no caso foram 10 execuções.

## Sobre o dataset

Link do dataset: [Taxi Price Prediction (Kaggle)](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction)

A variável alvo (target) do problema é a coluna `Trip_Price`.

## Pipeline do benchmark

O fluxo completo acontece em três etapas, cada uma isolada em um módulo:

### 1. Extração e pré-processamento (`pipeline.py`)

1. **Leitura**: o CSV é carregado com `pandas.read_csv`.
2. **Limpeza do alvo**: linhas onde `Trip_Price` está ausente são descartadas.  
3. **Separação de features e alvo**: `X` recebe todas as colunas exceto `Trip_Price`; `y` recebe apenas `Trip_Price`.
4. **Identificação dos tipos de coluna**:
   - Colunas numéricas seguem para o pipeline numérico.
   - Colunas categóricas seguem para o pipeline categórico.
5. **Pipeline numérico**: valores ausentes são imputados pela mediana, em seguida os valores são normalizados para o intervalo [0, 1].
6. **Pipeline categórico**: valores ausentes são imputados pelo valor mais frequente, em seguida as categorias são convertidas em variáveis binárias.
7. **Consolidação**: os dois pipelines são combinados em um único `ColumnTransformer`.
8. **Split treino/teste**: os dados são divididos em 80% treino / 20% teste.
9. **Ajuste e transformação**: o `ColumnTransformer` é ajustado apenas nos dados de treino, e só depois aplicado no teste.

> Observação: o mesmo pré-processamento é aplicado a todos os modelos, inclusive ao CatBoost, mesmo ele tendo suporte nativo a variáveis categóricas. Essa escolha foi feita propositalmente, para manter as condições de teste idênticas entre os três frameworks.

### 2. Execução do benchmark (`metrics.py`)

Para cada modelo, a função `benchmark_metrics` roda 10 execuções independentes e em cada uma:

1. Uma instância nova do modelo é criada, para garantir que nenhum estado de uma rodada vaze para a próxima.
2. `gc.collect()` é chamado antes de medir, para reduzir ruído de lixo acumulado de rodadas anteriores.
3. O uso de memória do processo (RSS) é registrado antes do treino.
4. O treino é executado e cronometrado.
5. O uso de memória do processo (RSS) é registrado novamente após o treino, e a diferença entre os dois valores é usada como uso de memória da rodada.
6. A inferência no conjunto de teste é cronometrada separadamente.

Ao final das 10 execuções, as métricas de tempo e memória são consolidadas pela média, e as métricas estatísticas (MAE, RMSE, R²) são calculadas sobre as previsões da última rodada.

### 3. Consolidação dos resultados (`main.py`)

Os três modelos são instanciados com os mesmos parâmetros de paralelismo (4 threads), o pipeline é executado uma única vez para gerar os dados de treino/teste e cada modelo passa pela função de benchmark. 
Os resultados de todos os modelos são consolidados e exportados para `results.csv`.

## Métricas avaliadas

### Estatísticas

| Métrica | O que significa |
|---|---|
| **MAE** (Mean Absolute Error) | Erro médio absoluto entre valor previsto e valor real, na mesma unidade do target (aqui, preço da corrida). Quanto menor, melhor. |
| **RMSE** (Root Mean Squared Error) | Parecido com o MAE, mas penaliza mais os erros grandes. Útil para saber se o modelo comete erros grandes ocasionais. Quanto menor, melhor. |
| **R² Score** | Proporção da variância dos dados que o modelo consegue explicar, de 0 a 1 (pode ser negativo em modelos ruins). Quanto mais perto de 1, melhor o ajuste. |

### Computacionais

| Métrica | O que significa |
|---|---|
| **Avg Train Time (s)** | Tempo médio para treinar o modelo do zero, em segundos, ao longo das 10 execuções. |
| **Avg Infer Latency (ms)** | Tempo médio para gerar previsões no conjunto de teste, em milissegundos. |
| **Avg Peak RAM - RSS delta (MB)** | Variação da memória física (RSS) do processo entre antes e depois do treino. Captura o consumo real do processo, incluindo o que é alocado fora do Python. |

## Estrutura do projeto

```
.
├── data/
│   └── taxi_trip_pricing.csv   # dataset
├── utils/
│   ├── pipeline.py             # extração e pré-processamento dos dados
│   └── metrics.py              # execução do benchmark e consolidação de métricas
├── main.py                     # orquestra o benchmark e gera results.csv
├── results.csv                 # saída do benchmark (gerado ao rodar main.py)
├── pyproject.toml
└── README.md
```

## Como executar

O projeto usa [uv](https://docs.astral.sh/uv/) como gerenciador de pacotes.
1. Instale as dependências:
   ```bash
   uv sync
   ```
2. Rode o benchmark:
   ```bash
   uv run main.py
   ```
3. Os resultados consolidados serão salvos em `results.csv`.

## Limitações conhecidas

- O benchmark foi executado localmente, em uma única máquina — os valores absolutos de tempo e memória não são portáveis para outro hardware; use-os para comparação **relativa** entre os modelos, não como número absoluto de referência.
- Os modelos foram comparados com os **hiperparâmetros padrão** de cada biblioteca, não com hiperparâmetros otimizados, logo os resultados refletem o comportamento "out-of-the-box" de cada framework, não o teto de performance possível de cada um.
- O CatBoost não teve acesso ao seu tratamento nativo de variáveis categóricas, para manter o pré-processamento idêntico entre os três modelos.
