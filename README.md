# Ensemble Pipeline (SA/MA) — Guia de Uso

Este documento explica como usar o pipeline de ensemble por modelo (SA) e entre modelos (MA), como preparar o ambiente, onde os resultados são salvos, como os dados devem estar estruturados e fornece uma visão da estrutura do projeto.

## Visão Geral
- SA (per-model ensemble): combina predições de múltiplos folds/execuções do MESMO modelo para gerar artefatos nos níveis tile, imagem e paciente.
- MA (between-models ensemble): combina os artefatos dos modelos (já consolidados via SA) para produzir resultados agregados entre arquiteturas.
- Gráficos (ROC, matriz de confusão, classification report) são gerados seguindo o estilo de `all_metrics_generate.py` e ficam copiados em `outputs/results/...`.

## Estrutura do Projeto (diagrama)
```
meu cod/
├── main.py                         # Funções de descoberta de dados, execução e exportação (SA/MA)
├── modules/
│   ├── per_model_ensembler.py      # Lógica do ensemble por modelo (SA) + cache interno
│   ├── ensemble_between_models.py  # Lógica do ensemble entre modelos (MA)
│   ├── runner_sa.py                # Pipeline SA com paralelização e flags
│   ├── utils.py                    # Funções de plot (ROC, Confusion, Report) e utilidades
│   ├── flags.py                    # Flags centrais (níveis, tipos de plots, paralelização)
│   └── __init__.py                 # Exposição das classes principais
├── all_metrics_generate.py         # Template visual (referência de estilo para plots)
├── Emsemble_pipline_models_optimized.ipynb  # Notebook otimizado para rodar SA/MA
├── notebooks/                      # Notebooks originais e pastas de resultados por modelo
└── outputs/
    └── tables/                     # CSV/JSON de SA/MA por nível e por modelo
        └── Ensemble_Between_Models # Artefatos consolidados do MA
    └── results/                    # Cópias + plots prontos (ROC/Confusion/Report)
```

## Preparar Ambiente de Execução
Recomendado usar Conda no Windows.

1. Instale [Miniconda/Anaconda](https://www.anaconda.com/).
2. No PowerShell, dentro da pasta do projeto:
   - `conda env create -f environment.yml`
   - `conda activate seu_ambiente`  (o nome é o definido em `environment.yml`)
3. (Opcional) Instale Jupyter:
   - `pip install jupyterlab`
   - iniciar: `jupyter lab` ou `jupyter notebook`

## Estrutura dos Dados (entrada)
- O pipeline procura CSVs de folds/resultados por modelo nas seguintes pastas, em ordem:
  1. `datas/<summary_results_model>/` (espelho automático de `notebooks/`)
  2. `notebooks/<summary_results_model>/`
  3. Busca recursiva por padrões contendo o nome do modelo e `fold`

- Pastas típicas em `notebooks/` (por modelo):
  - `summary_results_ggnet`, `summary_results_effnet`, `summary_results_mobnet`, etc.

- Colunas mínimas esperadas em cada CSV:
  - `true_label` (rótulo verdadeiro)
  - `predicted_label` (rótulo por argmax)
  - Para plots de ROC com soft/weighted: um vetor de probabilidades por classe. O utilitário tenta localizar uma destas colunas:
    - `final_probs` (string representando dict Python, ex.: `{'classA': 0.2, 'classB': 0.8}`)
    - `mean_probs_per_class` ou `mean_probs_per_class_<level>`

Observação: Se apenas `predicted_label` existir, os gráficos de ROC serão gerados no modo hard voting (one-vs-rest via rótulos discretos).

## Onde os Resultados São Salvos
O pipeline grava artefatos (CSV/JSON) em `outputs/tables/` e faz uma cópia organizada com gráficos em `outputs/results/`.

- SA (por modelo):
  - `outputs/tables/<MODEL>/Ensemble_tile_level_<tipo>/ensemble_per_tile_<tipo>.csv`
  - `outputs/tables/<MODEL>/Ensemble_image_level_<tipo>/ensemble_per_image_<tipo>.csv`
  - `outputs/tables/<MODEL>/Ensemble_patient_level_<tipo>/ensemble_per_patient_<tipo>.csv`
  - `outputs/results/SA/<MODEL>/<tipo>/<Level>/` (CSV, JSON e PNGs — ROC/Confusion/Report)
  - Para `weighted`, o sufixo inclui a métrica de peso, ex.: `weighted_f1_macro`.

- MA (entre modelos):
  - `outputs/tables/Ensemble_Between_Models/<Level>/<tipo>/` (CSV/JSON consolidados)
  - `outputs/results/MA/<tipo>/<Level>/` (cópias + PNGs)

## Formas de Execução

### A) Notebook Otimizado (recomendado)
Abra `Emsemble_pipline_models_optimized.ipynb` e siga as células:

1. Defina as flags:
   - Em `modules/flags.py` ou diretamente no notebook:
     - `GENERATE_LEVELS = ['tile', 'image', 'patient']` (escolha níveis)
     - `GENERATE_PLOTS_FOR = ['soft_voting']` (opções: `hard_voting`, `soft_voting`, `weighted`)
     - `ROC_DETAIL = 'per_class'` ou `'macro_micro'`
     - `MAX_WORKERS` e `USE_THREADS` (Windows → `USE_THREADS=True`)
2. Configure a lista de modelos e rode SA:
   - `MODELS = ['EFFNet', 'GGNet', 'MOBNet']`
   - `results = run_sa_pipeline(models_cfgs)`
3. (Opcional) Rode MA:
   - `run_between_models(MODELS, ensemble_type='soft_voting', weight_metric='f1_macro')`
4. Verifique os artefatos e gráficos em `outputs/results/...`.

### B) Script/Python
Exemplo mínimo para MA (garantindo SA automaticamente):
```python
from main import run_between_models

MODELS = ['EFFNet', 'GGNet', 'MOBNet']
run_between_models(MODELS, ensemble_type='soft_voting', weight_metric='f1_macro')
```
Isso irá:
1. Conferir se os artefatos de SA existem para cada modelo.
2. Gerar SA em paralelo para os modelos faltantes.
3. Executar MA nos níveis imagem, tile e paciente.
4. Exportar tudo para `outputs/results/MA/...` com gráficos.

## Tipos de Ensemble e Plots
- `hard_voting`: voto majoritário baseado em `predicted_label`.
- `soft_voting`: média (ou média ponderada) dos vetores de probabilidade.
- `weighted`: aplica pesos por fonte (fold/modelo), afetando hard e soft.

Plots gerados automaticamente:
- ROC: `roc_curve_model_<level>_level_<tipo>.png`
- Matriz de Confusão: `confusion_matrix_model_<level>_level_ensemble.png`
- Classification Report: `classification_report_model_<level>_level_ensemble.png`

## Fluxo Interno (resumo)
1. SA (modules/per_model_ensembler.py):
   - Consolida por nível: tiles → imagem → paciente.
   - Evita recomputações com cache interno entre níveis.
   - Salva CSV/JSON e, via `runner_sa.py`, gera plots.
2. MA (modules/ensemble_between_models.py via main.py):
   - Usa artefatos de SA já disponíveis por modelo.
   - Consolida entre modelos e exporta para `outputs/results/MA/...` com plots.

## Problemas Comuns e Dicas
- Se algum gráfico não aparecer:
  - Verifique se o CSV possui `true_label` e `predicted_label`.
  - Para ROC em soft/weighted, confirme a coluna de probabilidades (`final_probs`, `mean_probs_per_class` ou `mean_probs_per_class_<level>`).
- Se não encontrar CSVs:
  - Coloque os resultados dos folds em `notebooks/summary_results_<modelo>/`.
  - Ou copie para `datas/summary_results_<modelo>/` (há uma função que espelha notebooks→datas).
- No Windows, use `USE_THREADS=True` para paralelização.

## Extensões
- Ajuste pesos do `weighted` com métricas robustas (ex.: `f1_macro`) em validação externa.
- Integre novos modelos adicionando a pasta `summary_results_<modelo>` com CSVs.
- Personalize plots (cores, títulos) editando `modules/utils.py`.

---
Para suporte adicional ou integração em notebooks existentes, abra uma issue ou peça para aplicar o fluxo otimizado diretamente no seu notebook atual.

## Exemplos práticos

### Exemplo de CSV mínimo (2 classes)
Cada linha representa uma unidade no nível corrente (tile ou imagem). Para tiles, inclua `image_id`; para imagens, inclua `patient_id`.

```
tile_id,image_id,patient_id,true_label,predicted_label,final_probs
tile_001,img_A,pat_X,ClassB,ClassB,"{'ClassA': 0.30, 'ClassB': 0.70}"
tile_002,img_A,pat_X,ClassB,ClassB,"{'ClassA': 0.10, 'ClassB': 0.90}"
tile_003,img_A,pat_X,ClassB,ClassA,"{'ClassA': 0.60, 'ClassB': 0.40}"
```

Observações:
- `final_probs` é uma string com dict por classe (o utilitário faz a conversão automática).
- Em hard voting, apenas `predicted_label` é necessário; em soft/weighted, use o vetor de probabilidades.

### Exemplo numérico — agregação de tiles → imagem
Dado o CSV acima, para `img_A`:
- Soft (média de probabilidades por classe):
  - P_image('ClassA') = (0.30 + 0.10 + 0.60) / 3 = 0.33
  - P_image('ClassB') = (0.70 + 0.90 + 0.40) / 3 = 0.67
  - Classe final: ClassB; Incerteza = 1 - max(P_image) = 1 - 0.67 = 0.33
- Hard (voto por rótulo):
  - Votos: ClassB (2), ClassA (1) → Classe final: ClassB
  - Vetor de contagem normalizada: P_image = [1/3, 2/3]; Incerteza = 1 - 2/3 = 0.33
- Weighted (ex.: pesos w=[0.5, 0.3, 0.2] por tile):
  - Soft ponderado: P_image = 0.5*[0.3,0.7] + 0.3*[0.1,0.9] + 0.2*[0.6,0.4] = [0.33, 0.67]
  - Hard ponderado: W_ClassB = 0.5 + 0.3 = 0.8; W_ClassA = 0.2 → Classe final: ClassB; P_image = [0.2, 0.8]

### Exemplo de execução (script)
```python
from main import run_between_models

MODELS = ['EFFNet', 'GGNet', 'MOBNet']
run_between_models(MODELS, ensemble_type='soft_voting', weight_metric='f1_macro')
```
Resultado:
1) Gera SA automaticamente onde faltar.
2) Executa MA nos níveis imagem, tile e paciente.
3) Copia CSV/JSON e salva gráficos em `outputs/results/MA/<tipo>/<Level>/`.

### Exemplo de execução (notebook)
No `Emsemble_pipline_models_optimized.ipynb`, ajuste as flags e rode:
```python
import modules.flags as flags
from modules.runner_sa import run_sa_pipeline
from main import discover_model_csvs, resolve_paths_outputs, run_between_models

flags.GENERATE_LEVELS = ['image','patient']
flags.GENERATE_PLOTS_FOR = ['soft_voting']
flags.ROC_DETAIL = 'macro_micro'

MODELS = ['EFFNet','GGNet']
tables_dir, _ = resolve_paths_outputs()
models_cfgs = []
for m in MODELS:
    csvs = discover_model_csvs(m)
    models_cfgs.append({'model_name': m, 'ensemble_type': 'soft_voting', 'csv_paths': csvs, 'save_output_base': os.path.join(tables_dir, m)})

results = run_sa_pipeline(models_cfgs)
run_between_models(MODELS, ensemble_type='soft_voting', weight_metric='f1_macro')
```

### Onde encontrar os gráficos
- SA: `outputs/results/SA/<MODEL>/<tipo>/<Level>/` → `roc_curve_*.png`, `confusion_matrix_*.png`, `classification_report_*.png`
- MA: `outputs/results/MA/<tipo>/<Level>/` → mesmos nomes de arquivos.

### Troubleshooting rápido
- Erro para abrir notebook: verifique se o arquivo `.ipynb` tem JSON válido (cada `source` deve ser lista de strings). O notebook otimizado já foi corrigido.
- Sem gráficos de ROC em soft/weighted: confirme a presença de `final_probs` (ou `mean_probs_per_class_*`).
- SA não encontrado para um modelo: cheque `notebooks/summary_results_<modelo>/` ou copie para `datas/summary_results_<modelo>/`.