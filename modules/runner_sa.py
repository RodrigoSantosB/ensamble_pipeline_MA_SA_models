import os
from typing import List, Dict

from .per_model_ensembler import PerModelEnsembler
from .utils import plot_roc_from_csv, save_confusion_and_report_from_csv, run_in_parallel
from .flags import GENERATE_LEVELS, GENERATE_PLOTS_FOR, ROC_DETAIL, MAX_WORKERS, USE_THREADS


def _run_sa_for_model(model_cfg: Dict) -> Dict:
    """Executa SA para um único modelo conforme flags configuradas.
    model_cfg: {
        'model_name': str,
        'ensemble_type': 'hard_voting'|'soft_voting'|'weighted_voting',
        'csv_paths': List[str],
        'save_output_base': Optional[str]
    }
    Retorna dict com paths gerados por nível.
    """
    model_name = model_cfg['model_name']
    ensemble_type = model_cfg.get('ensemble_type', 'hard_voting')
    csv_paths = model_cfg['csv_paths']
    save_output_base = model_cfg.get('save_output_base')

    sa = PerModelEnsembler(model_name=model_name, ensemble_type=ensemble_type, save_output_base=save_output_base)
    results = {}

    if 'tile' in GENERATE_LEVELS:
        tile_csv, out_dir = sa.run_tile_level(csv_paths)
        results['tile'] = {'csv': tile_csv, 'out_dir': out_dir}
        if ensemble_type in GENERATE_PLOTS_FOR:
            plot_roc_from_csv(tile_csv, ensemble_type, out_dir, level='tile', network=model_name, detail=ROC_DETAIL)
            save_confusion_and_report_from_csv(tile_csv, out_dir, level='tile', network=model_name)

    if 'image' in GENERATE_LEVELS:
        img_csv, out_dir = sa.run_image_level(csv_paths)
        results['image'] = {'csv': img_csv, 'out_dir': out_dir}
        if ensemble_type in GENERATE_PLOTS_FOR:
            plot_roc_from_csv(img_csv, ensemble_type, out_dir, level='image', network=model_name, detail=ROC_DETAIL)
            save_confusion_and_report_from_csv(img_csv, out_dir, level='image', network=model_name)

    if 'patient' in GENERATE_LEVELS:
        pat_csv, out_dir = sa.run_patient_level(csv_paths)
        results['patient'] = {'csv': pat_csv, 'out_dir': out_dir}
        if ensemble_type in GENERATE_PLOTS_FOR:
            plot_roc_from_csv(pat_csv, ensemble_type, out_dir, level='patient', network=model_name, detail=ROC_DETAIL)
            save_confusion_and_report_from_csv(pat_csv, out_dir, level='patient', network=model_name)

    return results


def run_sa_pipeline(models_cfgs: List[Dict]) -> List[Dict]:
    """Executa SA em paralelo para lista de modelos.
    models_cfgs: lista de model_cfg conforme _run_sa_for_model.
    Retorna lista de resultados na mesma ordem.
    """
    tasks = [(_run_sa_for_model, {'model_cfg': cfg}) for cfg in models_cfgs]
    return run_in_parallel(tasks, max_workers=MAX_WORKERS, use_threads=USE_THREADS)