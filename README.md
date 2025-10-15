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

Notas:
- `modules/flags.py` centraliza níveis, tipos de plots e paralelismo.
- `modules/metrics.py` cuida de métricas e gráficos a partir de CSVs.
- `modules/utils.py` oferece descoberta de dados e utilitários.
- `main.py` orquestra SA e MA exclusivamente via funções dos módulos.

---
Para dúvidas, abra uma issue ou solicite orientação para adaptar o pipeline ao seu conjunto de dados.
