"""CLI de orquestração do pipeline de ensemble (SA e MA).

Este módulo expõe uma interface de linha de comando para executar o
pipeline de Single-Model Aggregation (SA) e Multi-Model Aggregation (MA),
incluindo geração de métricas e gráficos. As operações são realizadas
utilizando funções utilitárias e runners em `modules/*`.

Uso básico:
    python main.py --models "EFFNet,GGNet" --sa-type weighted --ma-type soft_voting

"""
import argparse
import os
from typing import List, Dict

# Logs and discovery
from modules.utils import log, discover_models_and_paths, resolve_paths_outputs

# SA and MA runners (purely using module implementations)
from modules.runner_sa import run_sa_for_models_parallel
from modules.runner_ma import run_ma_for_models_parallel

# Metrics/plots generation helpers (write to outputs/results)
from modules.metrics import generate_sa_metrics_from_tables, generate_ma_metrics_from_tables

# Control flags
from modules.flags import MAX_WORKERS, USE_THREADS


def _parse_args():
    """Cria e parseia os argumentos de CLI para o pipeline.

    Returns:
        argparse.Namespace: Estrutura contendo opções de execução do pipeline,
            incluindo seleção de modelos, tipos e métricas de peso para SA/MA,
            controle de paralelismo, níveis e flags para pular etapas.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline principal para executar ensembles SA/MA e gerar métricas/plots usando apenas funções dos módulos."
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Lista de modelos separados por vírgula (ex: 'EFFNet,GGNet'). Se omitido, detecta automaticamente em 'datas/'.",
    )

    # SA controls
    parser.add_argument(
        "--sa-type",
        type=str,
        default="soft_voting",
        choices=["hard_voting", "soft_voting", "weighted"],
        help="Tipo de ensemble para SA (por modelo).",
    )
    parser.add_argument(
        "--sa-weight-metric",
        type=str,
        default="f1_macro",
        help="Métrica para ponderação no SA quando '--sa-type=weighted' (ex: f1_macro, accuracy, recall_weighted, roc_auc_ovr).",
    )
    parser.add_argument(
        "--skip-sa",
        action="store_true",
        help="Pula a execução do pipeline SA (por modelo).",
    )

    # MA controls
    parser.add_argument(
        "--ma-type",
        type=str,
        default="soft_voting",
        choices=["hard_voting", "soft_voting", "weighted"],
        help="Tipo de ensemble para MA (entre modelos).",
    )
    parser.add_argument(
        "--ma-weight-metric",
        type=str,
        default="f1_macro",
        help="Métrica para ponderação no MA quando '--ma-type=weighted' (ex: f1_macro, accuracy, recall_weighted, roc_auc_ovr).",
    )
    parser.add_argument(
        "--skip-ma",
        action="store_true",
        help="Pula a execução do pipeline MA (entre modelos).",
    )

    # Parallelism controls
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Número máximo de workers para execução paralela (default vem de modules.flags).",
    )
    parser.add_argument(
        "--use-threads",
        action="store_true",
        default=USE_THREADS,
        help="Usa threads para paralelismo (Windows-friendly).",
    )

    return parser.parse_args()


def _run_sa_pipeline(models_map: Dict[str, List[str]], sa_type: str, sa_weight_metric: str, tables_dir: str, max_workers: int, use_threads: bool) -> None:
    """Executa o pipeline SA (por modelo) para os níveis configurados.

    Agrega resultados por modelo nos níveis `tile`, `image` e `patient`,
    invocando o runner responsável por paralelizar o processamento por
    modelo e nível. Após gerar artefatos (tables), também aciona a
    geração de métricas e gráficos (results).

    Args:
        models_map (dict[str, list[str]]): Mapeamento de nome do modelo para
            lista de caminhos de CSVs de entrada (folds/execuções) esperados.
        sa_type (str): Tipo de ensemble por modelo. Um de
            {"hard_voting", "soft_voting", "weighted"}.
        sa_weight_metric (str): Métrica para ponderação no modo `weighted`
            (ex.: "f1_macro", "accuracy"). Ignorado para outras modalidades.
        tables_dir (str): Caminho do diretório base de `outputs/tables`.
        max_workers (int): Número máximo de workers para paralelização.
        use_threads (bool): Se verdadeiro, utiliza threads; caso contrário,
            utiliza processos.

    Returns:
        None: Executa com efeitos colaterais, salvando artefatos e métricas
            em diretórios `outputs/tables` e `outputs/results` conforme
            configuração.
    """
    if not models_map:
        log("[SA] Nenhum modelo encontrado para execução.")
        return

    # Transforma o dict em lista de configs conforme runner_sa espera
    loaded_configs = []
    for model_name, csv_paths in models_map.items():
        if not csv_paths:
            log(f"[SA] {model_name}: sem CSVs, pulando.")
            continue
        loaded_configs.append({
            "model_name": model_name,
            "ensemble_type": sa_type,
            "weight_metric": sa_weight_metric,
            "csv_paths": csv_paths,
        })

    if not loaded_configs:
        log("[SA] Nenhuma configuração válida para execução.")
        return

    # Executa SA paralelamente por modelo
    run_sa_for_models_parallel(
        TABLES_DIR=tables_dir,
        loaded_configs=loaded_configs,
        max_workers=max_workers,
        use_threads=use_threads,
    )

    # Gera métricas e plots para SA diretamente em outputs/results/SA/<MODEL>/<LEVEL>
    try:
        generate_sa_metrics_from_tables(models=list(models_map.keys()))
    except Exception as e:
        log(f"[SA] Falha ao gerar métricas/plots SA a partir de tables: {e}")


def _run_ma_pipeline(models_list: List[str], ma_type: str, ma_weight_metric: str, tables_dir: str, max_workers: int, use_threads: bool) -> None:
    """Executa o pipeline MA (entre modelos) para os níveis configurados.

    Combina artefatos provenientes do SA por modelo para gerar ensemble
    entre modelos nos níveis `tile`, `image` e `patient`. Após gerar
    artefatos (tables), também aciona a geração de métricas e gráficos.

    Args:
        models_list (list[str]): Lista com os nomes dos modelos a considerar
            no ensemble entre modelos.
        ma_type (str): Tipo de ensemble entre modelos. Um de
            {"hard_voting", "soft_voting", "weighted"}.
        ma_weight_metric (str): Métrica de ponderação no modo `weighted`
            (ex.: "f1_macro", "accuracy"). Ignorado nas demais modalidades.
        tables_dir (str): Caminho do diretório base de `outputs/tables`.
        max_workers (int): Número máximo de workers para paralelização.
        use_threads (bool): Se verdadeiro, utiliza threads; caso contrário,
            utiliza processos.

    Returns:
        None: Executa com efeitos colaterais, salvando artefatos e métricas
            nos diretórios `outputs/tables` e `outputs/results`.
    """
    if not models_list:
        log("[MA] Lista de modelos vazia, pulando MA.")
        return

    run_ma_for_models_parallel(
        tables_dir=tables_dir,
        models_to_include=models_list,
        ensemble_type=ma_type,
        weight_metric=ma_weight_metric,
        max_workers=max_workers,
        use_threads=use_threads,
    )

    # Gera métricas e plots para MA diretamente em outputs/results/MA/<LEVEL>
    try:
        generate_ma_metrics_from_tables()
    except Exception as e:
        log(f"[MA] Falha ao gerar métricas/plots MA a partir de tables: {e}")


def main():
    """Ponto de entrada para executar SA e/ou MA e gerar métricas.

    Lê parâmetros de CLI, resolve caminhos de dados/saídas e, conforme
    flags e opções fornecidas, orquestra a execução dos pipelines SA e MA,
    além da geração de métricas e gráficos agregados.

    Returns:
        None
    """
    args = _parse_args()

    # Resolve diretórios de saída padrão (outputs/tables e subpastas)
    tables_dir, _ = resolve_paths_outputs()
    log(f"[PATHS] Tabelas em: {tables_dir}")

    # Detecta modelos e CSVs de folds em 'datas/'
    models_requested = None
    if args.models:
        models_requested = [m.strip() for m in args.models.split(',') if m.strip()]
    models_map = discover_models_and_paths(models_requested)
    models_list = list(models_map.keys())
    log(f"[DISCOVERY] Modelos prontos para execução: {models_list}")

    # SA pipeline
    if not args.skip_sa:
        log(f"[SA] Executando SA com tipo='{args.sa_type}' e weight_metric='{args.sa_weight_metric}'")
        _run_sa_pipeline(
            models_map=models_map,
            sa_type=args.sa_type,
            sa_weight_metric=args.sa_weight_metric,
            tables_dir=tables_dir,
            max_workers=args.max_workers,
            use_threads=args.use_threads,
        )
    else:
        log("[SA] Pulado por opção --skip-sa")

    # MA pipeline
    if not args.skip_ma:
        log(f"[MA] Executando MA com tipo='{args.ma_type}' e weight_metric='{args.ma_weight_metric}'")
        _run_ma_pipeline(
            models_list=models_list,
            ma_type=args.ma_type,
            ma_weight_metric=args.ma_weight_metric,
            tables_dir=tables_dir,
            max_workers=args.max_workers,
            use_threads=args.use_threads,
        )
    else:
        log("[MA] Pulado por opção --skip-ma")

    log("--- Pipeline concluído ---")


if __name__ == "__main__":
    main()