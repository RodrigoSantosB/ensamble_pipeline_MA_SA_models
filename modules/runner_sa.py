"""Execução orquestrada do pipeline SA (por modelo) nos três níveis.

Define helpers para determinar diretórios de saída por nível, executar cada
nível de ensemble de um `PerModelEnsembler` e paralelizar a execução por
modelo, salvando artefatos e métricas.
"""
import os
from typing import List, Dict, Tuple
from .per_model_ensembler import PerModelEnsembler
from modules.common import sanitize_out_dir, unlock_path, run_in_parallel
from modules.utils import log




def _out_dir_for(sa: 'PerModelEnsembler', level: str) -> Tuple[str, str, str]:
    """Resolve diretório e nomes de arquivos por nível.

    Args:
        sa (PerModelEnsembler): Instância configurada para o modelo.
        level (str): Nível (`tile`, `image`, `patient`).

    Returns:
        Tuple[str, str, str]: `(out_dir, csv_name, metrics_name)` para salvar
            artefatos e métricas no nível especificado.

    Raises:
        ValueError: Se `level` não for reconhecido.
    """
    if level == 'tile':
        if sa.ENSEMBLE_TYPE == 'weighted':
            subfolder, csv_name, metrics_name = 'Ensemble_tile_level_weighted', f'ensemble_per_tile_weighted_{sa.WEIGHT_METRIC}.csv', f'global_metrics_tile_level_weighted_{sa.WEIGHT_METRIC}.json'
        else:
            subfolder, csv_name, metrics_name = f'Ensemble_tile_level_{sa.ENSEMBLE_TYPE}', f'ensemble_per_tile_{sa.ENSEMBLE_TYPE}.csv', f'global_metrics_tile_level_{sa.ENSEMBLE_TYPE}.json'
    elif level == 'image':
        if sa.ENSEMBLE_TYPE == 'weighted':
            subfolder, csv_name, metrics_name = 'Ensemble_image_level_weighted', f'ensemble_per_image_weighted_{sa.WEIGHT_METRIC}.csv', 'ensemble_global_metrics_image_level_weighted.json'
        else:
            subfolder, csv_name, metrics_name = f'Ensemble_image_level_{sa.ENSEMBLE_TYPE}', f'ensemble_per_image_{sa.ENSEMBLE_TYPE}.csv', f'ensemble_global_metrics_image_level_{sa.ENSEMBLE_TYPE}.json'
    elif level == 'patient':
        if sa.ENSEMBLE_TYPE == 'weighted':
            subfolder, csv_name, metrics_name = 'Ensemble_patient_level_weighted', f'ensemble_per_patient_weighted_{sa.WEIGHT_METRIC}.csv', f'global_metrics_patient_level_weighted_{sa.WEIGHT_METRIC}.json'
        else:
            subfolder, csv_name, metrics_name = f'Ensemble_patient_level_{sa.ENSEMBLE_TYPE}', f'ensemble_per_patient_{sa.ENSEMBLE_TYPE}.csv', f'global_metrics_patient_level_{sa.ENSEMBLE_TYPE}.json'
    else:
        raise ValueError(f'Nível desconhecido: {level}')
    out_dir = os.path.join(sa.save_output_base, subfolder)
    return out_dir, csv_name, metrics_name



def _run_sa_for_model(sa: 'PerModelEnsembler', level: str, csv_paths: List[str]) -> Tuple[str, str]:
    """Executa o ensemble SA para um nível específico de um modelo.

    Prepara diretório e nomes de saída, desbloqueia caminhos quando
    necessário e invoca `run_<level>_level` na instância `PerModelEnsembler`.

    Args:
        sa (PerModelEnsembler): Instância do ensembler por modelo.
        level (str): Nível (`tile`, `image`, `patient`).
        csv_paths (List[str]): Caminhos de CSVs de folds/execuções.

    Returns:
        Tuple[str, str]: `(csv_path, metrics_path)` salvos para o nível.
    """
    out_dir, csv_name, metrics_name = _out_dir_for(sa, level)
    sanitize_out_dir(out_dir)
    unlock_path(os.path.join(out_dir, csv_name))
    unlock_path(os.path.join(out_dir, metrics_name))
    try:
        run_func = getattr(sa, f'run_{level}_level')
        return run_func(csv_paths)
    except Exception as e:
        msg = str(e)
        log(f'[AVISO] Erro no nível {level}: {msg}')
        if any(err in msg for err in ['WinError 183', 'FileExistsError', 'arquivo já existente', 'EEXIST', 'File exists']):
            sanitize_out_dir(out_dir)
            unlock_path(os.path.join(out_dir, csv_name))
            unlock_path(os.path.join(out_dir, metrics_name))
            try:
                run_func = getattr(sa, f'run_{level}_level')
                return run_func(csv_paths)
            except Exception as e2:
                log(f'[AVISO] Nível {level} falhou novamente: {e2}. Prosseguindo...')
        else:
            log(f'[AVISO] Nível {level} com erro inesperado. Prosseguindo.')
    return os.path.join(out_dir, csv_name), os.path.join(out_dir, metrics_name)


def _process_single_model(config: Dict, tables_dir: str) -> str:
    """Worker que executa o pipeline SA completo para um único modelo.

    Args:
        config (Dict): Configuração do modelo contendo `model_name`,
            `ensemble_type`, `weight_metric` e `csv_paths`.
        tables_dir (str): Diretório base de saída em `outputs/tables`.

    Returns:
        str: Mensagem resumindo o resultado da execução para o modelo.
    """
    model_name = config['model_name']
    ensemble_type = config['ensemble_type']
    weight_metric = config['weight_metric']
    csvs = config['csv_paths']  

    log(f'[INICIANDO] Processamento do modelo: {model_name}')
    
    # A verificação de CSVs vazios já é feita antes, mas mantemos por segurança
    if not csvs:
        log(f'[AVISO] Sem CSVs fornecidos para {model_name}. Pulando.')
        return f"{model_name}: Sem CSVs"

    save_base = os.path.join(tables_dir, model_name)
    sa = PerModelEnsembler(model_name=model_name, ensemble_type=ensemble_type, save_output_base=save_base)
    
    log(f'[SA] {model_name} → tile')
    tile_csv, tile_metrics = _run_sa_for_model(sa, 'tile', csvs)
    
    log(f'[SA] {model_name} → image')
    image_csv, image_metrics = _run_sa_for_model(sa, 'image', csvs)
    
    log(f'[SA] {model_name} → patient')
    patient_csv, patient_metrics = _run_sa_for_model(sa, 'patient', csvs)

    try:
        export_sa_to_results(tables_dir, model_name, ensemble_type, weight_metric, tile_csv, tile_metrics, image_csv, image_metrics, patient_csv, patient_metrics)
    except Exception as e:
        log(f'[AVISO] Falha ao exportar SA para {model_name}: {e}')

    log(f'[CONCLUÍDO] Processamento do modelo: {model_name}')
    return f"{model_name}: Concluído"


def run_sa_for_models_parallel(
    TABLES_DIR: str,
    loaded_configs: List[Dict],
    max_workers: int = 4,
    use_threads: bool = True
) -> None:
    """Executa o pipeline SA em paralelo para múltiplos modelos.

    Recebe configurações por modelo e dispara a execução paralela do
    ensemble `tile → image → patient`, gerando CSVs e métricas.

    Args:
        TABLES_DIR (str): Diretório base para salvar artefatos de saída.
        loaded_configs (List[Dict]): Lista de configurações por modelo.
        max_workers (int): Número máximo de workers.
        use_threads (bool): Se verdadeiro, usa threads; caso contrário, processos.

    Returns:
        None
    """
    if not loaded_configs:
        log("[INFO] A lista de configurações está vazia. Nenhuma tarefa para executar.")
        return

    num_tasks = len(loaded_configs)
    log(f"[SA] Iniciando execução do pipeline SA para {num_tasks} modelos (tile → image → patient)...")

    # Prepara a lista de tarefas a partir das configurações carregadas
    tasks = []
    for config_item in loaded_configs:
        # Cada item da lista já é o dicionário 'config' que a função worker precisa.
        # A tarefa é a tupla: (função_worker, {argumentos})
        tasks.append((_process_single_model, {'config': config_item, 'tables_dir': TABLES_DIR}))

    # Executa todas as tarefas em paralelo
    results = run_in_parallel(tasks, max_workers=max_workers, use_threads=use_threads)

    log("\n--- Pipeline SA Finalizado ---")
    log("Resultados das execuções:")
    for result in results:
        log(f" - {result}")
