"""Execução orquestrada do pipeline MA (entre modelos) nos três níveis.

Define função worker para processar um nível específico usando
`BetweenModelsEnsembler` e função orquestradora para paralelizar a
execução em `tile`, `image` e `patient`, salvando artefatos e métricas.
"""
import os
from typing import List, Dict, Tuple

# Importa a classe principal para o ensemble entre modelos
from modules.ensemble_between_models import BetweenModelsEnsembler
from modules.utils import log
from modules.common import run_in_parallel



# ==============================================================================
# FUNÇÃO WORKER PARA O ENSEMBLE MA
# ==============================================================================

def _process_ma_level(config: Dict) -> Tuple[str, str, str]:
    """Worker que executa o ensemble MA para um único nível.

    Instancia `BetweenModelsEnsembler` com os parâmetros fornecidos e
    chama `run_<level>_level`, retornando nível, caminho do CSV e caminho
    do JSON de métricas (ou mensagem de erro).

    Args:
        config (Dict): Configuração contendo:
            - `level` (str): Nível (`tile`, `image`, `patient`).
            - `ensemble_type` (str): Método (`hard_voting`, `soft_voting`, `weighted`).
            - `base_models_parent_directory` (str): Diretório base `outputs/tables`.
            - `ensemble_save_output_base` (str): Diretório para salvar resultados de MA.
            - `models_to_include` (List[str]): Modelos a incluir no ensemble.
            - `weight_metric` (str): Métrica de ponderação para `weighted`.

    Returns:
        Tuple[str, str, str]: `(level, csv_path, metrics_path)` no sucesso;
            em caso de erro, retorna `(level, "FALHA", <mensagem>)`.
    """
    # Desempacota a configuração
    level = config['level']
    ensemble_type = config['ensemble_type']
    base_dir = config['base_models_parent_directory']
    output_dir = config['ensemble_save_output_base']
    models = config['models_to_include']
    weight_metric = config['weight_metric']

    log(f"[MA INICIANDO] Executando ensemble '{ensemble_type}' para o nível '{level}'...")

    try:
        # Instancia o ensembler que combina resultados de diferentes modelos
        ensembler = BetweenModelsEnsembler(
            base_models_parent_directory=base_dir,
            ensemble_save_output_base=output_dir,
            models_to_include=models,
            ensemble_type=ensemble_type,
            weight_metric=weight_metric,
        )

        # Chama o método apropriado com base no nível (tile, image, patient)
        run_func = getattr(ensembler, f'run_{level}_level')
        csv_path, metrics_path = run_func()

        log(f"[MA SUCESSO] Nível '{level}' ({ensemble_type}) concluído. CSV salvo em: {csv_path}")
        return level, csv_path, metrics_path

    except Exception as e:
        log(f"[MA ERRO] Falha ao processar o nível '{level}' com ensemble '{ensemble_type}': {e}")
        return level, "FALHA", str(e)



# ==============================================================================
# FUNÇÃO PRINCIPAL ORQUESTRADORA DO PIPELINE MA
# ==============================================================================

# Dentro de modules/runner_ma.py

def run_ma_for_models_parallel(
    tables_dir: str,
    models_to_include: List[str],
    ensemble_type: str,
    weight_metric: str = 'f1_macro',
    max_workers: int = 3,
    use_threads: bool = True
) -> None:
    """Executa o pipeline MA em paralelo para os três níveis.

    Prepara tarefas para `tile`, `image` e `patient` utilizando o tipo
    de ensemble informado e dispara execução paralela.

    Args:
        tables_dir (str): Caminho base para `outputs/tables`.
        models_to_include (List[str]): Modelos que participarão do ensemble.
        ensemble_type (str): Método do ensemble (`hard_voting`, `soft_voting`, `weighted`).
        weight_metric (str): Métrica usada quando `ensemble_type='weighted'`.
        max_workers (int): Número máximo de workers.
        use_threads (bool): Se verdadeiro, usa threads; caso contrário, processos.

    Returns:
        None
    """
    if not models_to_include:
        log("[MA AVISO] A lista de modelos para incluir está vazia. Nenhuma tarefa para executar.")
        return

    log(f"[MA] Iniciando pipeline de Ensemble Entre Modelos para: {models_to_include}")
    log(f"[MA] Tipo de Ensemble a ser executado: '{ensemble_type}'")

    ensemble_output_base = os.path.join(tables_dir, 'Ensemble_Between_Models')
    os.makedirs(ensemble_output_base, exist_ok=True)
    log(f"[MA] Resultados serão salvos em: {ensemble_output_base}")

    # Prepara a lista de tarefas: uma para cada nível, usando o único tipo de ensemble fornecido
    tasks = []
    levels_to_run = ['tile', 'image', 'patient']

    # --- MUDANÇA: O loop sobre os tipos de ensemble foi removido ---
    for level in levels_to_run:
        config = {
            'level': level,
            'ensemble_type': ensemble_type, # Usa diretamente o parâmetro da função
            'base_models_parent_directory': tables_dir,
            'ensemble_save_output_base': ensemble_output_base,
            'models_to_include': models_to_include,
            'weight_metric': weight_metric,
        }
        tasks.append((_process_ma_level, {'config': config}))

    results = run_in_parallel(tasks, max_workers=max_workers, use_threads=use_threads)

    log("\n--- Pipeline MA (Ensemble Entre Modelos) Finalizado ---")
    log("Resultados das execuções:")
    for result in results:
        level, csv, status = result
        log(f" - Nível: {level:<8} | Status: {csv if csv == 'FALHA' else 'Sucesso'} | Detalhe: {status}")