"""Utilitários de arquivos, paralelismo e verificação de outputs SA/MA.

Este módulo fornece funções auxiliares para:
- Sanitizar diretórios de saída e desbloquear caminhos com permissões.
- Executar tarefas em paralelo via threads ou processos.
- Verificar e construir caminhos esperados para artefatos do pipeline SA.
- Checar pré-condições e acionar geração de outputs quando necessário.
"""
import os
from typing import Dict, List, Tuple
import shutil
import stat
from modules.utils import log



def _is_probable_file_name(name: str) -> bool:
    """Heurística simples para identificar nomes que parecem arquivos.

    Considera extensões comuns de artefatos (``.json``, ``.csv``, ``.png``,
    ``.jpg``, ``.jpeg``, ``.txt``). Útil para evitar diretórios com nomes
    conflitantes durante geração de arquivos.

    Args:
        name (str): Nome de arquivo ou diretório.

    Returns:
        bool: Verdadeiro se o nome aparenta ser de arquivo.
    """
    ext = os.path.splitext(name)[1].lower()
    return ext in {'.json', '.csv', '.png', '.jpg', '.jpeg', '.txt'}


def sanitize_out_dir(out_dir: str) -> None:
    """Sanitiza um diretório de saída para uso seguro.

    Garante que o diretório exista, remove pastas cujo nome sugere ser um
    arquivo (para evitar conflitos) e aplica permissões de escrita em
    arquivos existentes dentro do diretório.

    Args:
        out_dir (str): Caminho do diretório de saída.

    Returns:
        None
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        log(f'[AVISO] Não foi possível criar out_dir {out_dir}: {e}')
        return
    try:
        for entry in os.listdir(out_dir):
            path = os.path.join(out_dir, entry)
            if os.path.isdir(path) and _is_probable_file_name(entry):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    log(f'[SANIDADE] Removido diretório conflitante: {path}')
                except Exception as e:
                    log(f'[AVISO] Falha ao remover diretório conflitante {path}: {e}')
            elif os.path.isfile(path):
                try:
                    os.chmod(path, stat.S_IWRITE)
                except Exception:
                    pass
    except Exception as e:
        log(f'[AVISO] Não foi possível listar/inspecionar {out_dir}: {e}')


def unlock_path(path: str) -> None:
    """Garante permissão de escrita e remove diretórios conflitantes.

    Se ``path`` for um diretório cujo nome parece um arquivo (por exemplo,
    termina em ``.json``, ``.csv``, ``.png``), remove o diretório para
    evitar conflitos de criação de arquivo. Caso contrário, aplica
    permissão de escrita ao caminho informado.

    Args:
        path (str): Caminho de arquivo ou diretório a desbloquear.

    Returns:
        None
    """
    if not os.path.exists(path):
        return
    try:
        if os.path.isdir(path) and _is_probable_file_name(os.path.basename(path)):
            shutil.rmtree(path, ignore_errors=True)
            log(f'[SANIDADE] Removido diretório conflitante: {path}')
            return
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass


# ==============================================================================
# FUNÇÃO GENÉRICA PARA PARALELISMO 
# ==============================================================================
def run_in_parallel(callables_with_args: List[tuple], max_workers: int = 4, use_threads: bool = True):
    """Executa uma lista de tarefas em paralelo.

    Cada item em ``callables_with_args`` deve ser uma tupla ``(func, args_dict)``,
    onde ``func(**args_dict)`` será executada. Suporta execução com threads
    ou processos. Os resultados preservam a ordem de entrada.

    Args:
        callables_with_args (List[tuple]): Lista de tuplas ``(func, args_dict)``.
        max_workers (int): Número máximo de workers para o executor.
        use_threads (bool): Se verdadeiro, usa ``ThreadPoolExecutor``; caso
            contrário, usa ``ProcessPoolExecutor``.

    Returns:
        List: Lista de resultados na mesma ordem das tarefas.
    """
    results = []
    if use_threads:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(func, **kwargs): i for i, (func, kwargs) in enumerate(callables_with_args)}
            tmp = [None] * len(callables_with_args)
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    tmp[idx] = fut.result()
                except Exception as e:
                    tmp[idx] = e
            results = tmp
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(func, **kwargs): i for i, (func, kwargs) in enumerate(callables_with_args)}
            tmp = [None] * len(callables_with_args)
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    tmp[idx] = fut.result()
                except Exception as e:
                    tmp[idx] = e
            results = tmp
    return results


# ------------------------------
# Verificação de pré-condições SA antes de MA
# ------------------------------
def expected_sa_paths(tables_dir: str, model: str, ensemble_type: str, weight_metric: str) -> Dict[str, str]:
    """Constroi caminhos esperados dos artefatos SA para um modelo.

    Gera caminhos para CSVs e arquivos de métricas nos níveis ``tile``,
    ``image`` e ``patient``, considerando o tipo de ensemble e, se aplicável,
    a métrica de ponderação.

    Args:
        tables_dir (str): Diretório base de ``outputs/tables``.
        model (str): Nome do modelo (subpasta em ``tables_dir``).
        ensemble_type (str): Tipo de ensemble (``hard_voting``, ``soft_voting``, ``weighted``).
        weight_metric (str): Métrica usada quando ``ensemble_type='weighted'``.

    Returns:
        Dict[str, str]: Mapeamento de chave descritiva para caminho absoluto.
    """
    # Tile
    if ensemble_type == 'weighted':
        tile_sub = 'Ensemble_tile_level_weighted'
        tile_csv = f"ensemble_per_tile_weighted_{weight_metric}.csv"
        tile_metrics = f"global_metrics_tile_level_weighted_{weight_metric}.json"
    else:
        tile_sub = f"Ensemble_tile_level_{ensemble_type}"
        tile_csv = f"ensemble_per_tile_{ensemble_type}.csv"
        tile_metrics = f"global_metrics_tile_level_{ensemble_type}.json"
    tile_dir = os.path.join(tables_dir, model, tile_sub)

    # Image
    if ensemble_type == 'weighted':
        img_sub = 'Ensemble_image_level_weighted'
        img_csv = f"ensemble_per_image_weighted_{weight_metric}.csv"
        img_metrics = f"ensemble_global_metrics_image_level_weighted.json"
    else:
        img_sub = f"Ensemble_image_level_{ensemble_type}"
        img_csv = f"ensemble_per_image_{ensemble_type}.csv"
        img_metrics = f"ensemble_global_metrics_image_level_{ensemble_type}.json"
    img_dir = os.path.join(tables_dir, model, img_sub)

    # Patient
    if ensemble_type == 'weighted':
        pat_sub = 'Ensemble_patient_level_weighted'
        pat_csv = f"ensemble_per_patient_weighted_{weight_metric}.csv"
        pat_metrics = f"global_metrics_patient_level_weighted_{weight_metric}.json"
    else:
        pat_sub = f"Ensemble_patient_level_{ensemble_type}"
        pat_csv = f"ensemble_per_patient_{ensemble_type}.csv"
        pat_metrics = f"global_metrics_patient_level_{ensemble_type}.json"
    pat_dir = os.path.join(tables_dir, model, pat_sub)

    return {
        'tile_csv': os.path.join(tile_dir, tile_csv),
        'tile_metrics': os.path.join(tile_dir, tile_metrics),
        'image_csv': os.path.join(img_dir, img_csv),
        'image_metrics': os.path.join(img_dir, img_metrics),
        'patient_csv': os.path.join(pat_dir, pat_csv),
        'patient_metrics': os.path.join(pat_dir, pat_metrics),
    }


def check_sa_ready(tables_dir: str, model: str, ensemble_type: str, weight_metric: str) -> Tuple[bool, Dict[str, bool]]:
    """Verifica se os artefatos SA esperados existem para um modelo.

    Args:
        tables_dir (str): Diretório base de ``outputs/tables``.
        model (str): Nome do modelo.
        ensemble_type (str): Tipo de ensemble.
        weight_metric (str): Métrica para ponderação no modo ``weighted``.

    Returns:
        Tuple[bool, Dict[str, bool]]: ``all_ok`` e mapa de existência por artefato.
    """
    paths = expected_sa_paths(tables_dir, model, ensemble_type, weight_metric)
    status = {k: os.path.exists(v) for k, v in paths.items()}
    all_ok = all(status.values())
    return all_ok, status


def ensure_sa_outputs(models: List[str], ensemble_type: str, weight_metric: str) -> List[str]:
    """Garante geração/exportação dos artefatos SA para uma lista de modelos.

    Para cada modelo:
    - Verifica se todos os artefatos SA (``tile``, ``image``, ``patient``) existem.
    - Gera ensembles faltantes em paralelo.
    - Exporta SA para ``outputs/results`` quando completo.

    Args:
        models (List[str]): Lista de nomes de modelos.
        ensemble_type (str): Tipo de ensemble (``hard_voting``, ``soft_voting``, ``weighted``).
        weight_metric (str): Métrica para ponderação quando ``weighted``.

    Returns:
        List[str]: Modelos que ficaram prontos para uso no MA.
    """
    tables_dir, _ = resolve_paths_outputs()
    models_ready: List[str] = []
    to_generate_cfgs = []

    for m in models:
        ok, _ = check_sa_ready(tables_dir, m, ensemble_type, weight_metric)
        if ok:
            models_ready.append(m)
            continue
        # Descobrir CSVs de folds
        csvs = discover_model_csvs(m)
        if not csvs:
            print(f"[AVISO] Não encontrei CSVs de folds para o modelo {m}. Pulando este modelo no MA.")
            continue
        save_base = os.path.join(tables_dir, m)
        to_generate_cfgs.append({
            'model_name': m,
            'ensemble_type': 'weighted' if ensemble_type == 'weighted' else ensemble_type,
            'csv_paths': csvs,
            'save_output_base': save_base,
        })

    # Rodar SA em paralelo para os modelos pendentes
    if to_generate_cfgs:
        print(f"[INFO] Gerando ensembles SA em paralelo para {len(to_generate_cfgs)} modelo(s)...")
        _ = run_sa_pipeline(to_generate_cfgs)

    # Verificar novamente e exportar
    for m in models:
        ok2, _ = check_sa_ready(tables_dir, m, ensemble_type, weight_metric)
        if ok2:
            models_ready.append(m)
            export_sa_to_results(tables_dir, m, ensemble_type, weight_metric)
        else:
            print(f"[AVISO] SA incompleto para {m}, não será incluído no MA.")

    return models_ready
