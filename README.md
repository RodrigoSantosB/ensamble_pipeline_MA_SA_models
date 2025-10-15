# Pipeline de Ensemble (SA/MA)

Este projeto implementa um pipeline de ensemble em duas frentes: por modelo (SA) e entre modelos (MA), consolidando resultados nos níveis tile, imagem e paciente, e gerando métricas e gráficos prontos para análise.

## Visão Geral
- SA (por modelo): consolida predições de múltiplos folds/execuções do mesmo modelo em níveis `tile`, `image` e `patient`.
- MA (entre modelos): combina os artefatos dos diferentes modelos (já vindos do SA) para produzir resultados agregados.
- Métricas e gráficos (ROC, matriz de confusão, classification report) são salvos em `outputs/results/...` para consumo direto.

## Fluxo de Funcionamento
1. Descoberta de dados: o sistema localiza CSVs por modelo em `datas/summary_results_<modelo>/` (espelhados de `notebooks/`).
2. SA por modelo: `modules.runner_sa.run_sa_for_models_parallel` chama `modules.per_model_ensembler.PerModelEnsembler` para gerar conjuntos por nível e salvar em `outputs/tables/<MODEL>/Ensemble_<level>_level_<tipo>/`.
3. Métricas/plots SA: `modules.metrics.generate_sa_metrics_from_tables` varre `outputs/tables/<MODEL>` e salva métricas e gráficos em `outputs/results/SA/<MODEL>/<LEVEL>/`.
4. MA entre modelos: `modules.runner_ma.run_ma_for_models_parallel` lê as saídas de SA por modelo e gera CSVs e JSONs em `outputs/tables/Ensemble_Between_Models/<Level>_Ensemble_Models_<tipo>/`.
5. Métricas/plots MA: `modules.metrics.generate_ma_metrics_from_tables` copia e plota em `outputs/results/MA/<LEVEL>/`.

## Pré-requisitos
- Python 3.10+ e `pip`.
- Windows (testado) com PowerShell. Linux/Mac devem funcionar com ajustes mínimos.
- Ambiente Conda (recomendado) usando `environment.yml` ou um `venv` com dependências equivalentes.
- CSVs de entrada com colunas: `true_label`, `predicted_label` e (para soft/weighted) um vetor de probabilidades (`final_probs`, `mean_probs_per_class` ou `mean_probs_per_class_<level>`).

## Instalação
Opção Conda (recomendado):
- `conda env create -f environment.yml`
- `conda activate <nome_do_ambiente>`

Opção pip/venv (se tiver `requirements.txt`):
- `python -m venv .venv && .venv\Scripts\activate`
- `pip install -r requirements.txt`

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

## Como Usar
- Preparar a pasta de dados:
  - Crie a pasta `datas` e, para cada rede/modelo, a subpasta `datas\summary_results_<rede>\`.
  - Dentro dessa subpasta, salve os CSVs de cada fold seguindo o padrão `datas\summary_results_<rede>\<rede>_fold[i]_results.csv`.
  - Exemplo: `datas\summary_results_EFFNet\EFFNet_fold1_results.csv`, `datas\summary_results_EFFNet\EFFNet_fold2_results.csv`.
- Executar SA e MA com defaults:
  - `python main.py`
- Selecionar modelos específicos:
  - `python main.py --models "EFFNet,GGNet"`
- Controlar tipo de ensemble e métrica de peso:
  - SA: `python main.py --sa-type weighted --sa-weight-metric f1_macro`
  - MA: `python main.py --ma-type weighted --ma-weight-metric f1_macro`
- Pular etapas:
  - `python main.py --skip-sa` ou `python main.py --skip-ma`
- Ajustar paralelismo:
  - `python main.py --max-workers 4 --use-threads`

Resultados esperados:
- SA: `outputs/results/SA/<MODEL>/<LEVEL>/` com `metrics_*.json`, `roc_curve_*.png`, `confusion_matrix_*.png`, `classification_report_*.png`.
- MA: `outputs/results/MA/<LEVEL>/` com arquivos equivalentes.

## Estrutura do Projeto
```
ensamble_pipeline_MA_SA_models/
├── main.py
├── modules/
│   ├── common.py
│   ├── ensemble_between_models.py
│   ├── flags.py
│   ├── metrics.py
│   ├── per_model_ensembler.py
│   ├── runner_ma.py
│   ├── runner_sa.py
│   └── utils.py
├── Emsemble_pipline_models.ipynb
├── environment.yml
└── outputs/
    ├── tables/
    │   └── Ensemble_Between_Models/
    └── results/
```

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
from module.runner_ma import run_ma_for_models_parallel

TABLES_DIR = 'outputs/tables/Ensemble_Between_Models/'
ENSEMBLE_TYPE = 'soft_voting'
WEIGHT_METRIC = 'f1_macro'

MODELS = ['EFFNet', 'GGNet', 'MOBNet']

run_ma_for_models_parallel(
    tables_dir=TABLES_DIR,
    models_to_include=MODELS,
    ensemble_type=ENSEMBLE_TYPE, 
    weight_metric=WEIGHT_METRIC
)
```
Resultado:
1) Gera SA automaticamente onde faltar.
2) Executa MA nos níveis imagem, tile e paciente.
3) Copia CSV/JSON e salva gráficos em `outputs/results/MA/<tipo>/<Level>/`.

Notas:
- `modules/flags.py` centraliza níveis, tipos de plots e paralelismo.
- `modules/metrics.py` cuida de métricas e gráficos a partir de CSVs.
- `modules/utils.py` oferece descoberta de dados e utilitários.
- `main.py` orquestra SA e MA exclusivamente via funções dos módulos.

---
Para dúvidas, abra uma issue ou solicite orientação para adaptar o pipeline ao seu conjunto de dados.
