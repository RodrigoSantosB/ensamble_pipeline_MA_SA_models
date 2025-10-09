import os
import glob
import shutil
import json
import ast
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from modules import BetweenModelsEnsembler, PerModelEnsembler


def resolve_paths() -> tuple[str, str]:
    root = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.join(root, 'tables')
    ensemble_output_base = os.path.join(tables_dir, 'Ensemble_Between_Models')
    os.makedirs(ensemble_output_base, exist_ok=True)
    return tables_dir, ensemble_output_base


# ------------------------------
# Descoberta de CSVs por modelo (SA)
# ------------------------------
MODEL_SUMMARY_DIRS = {
    'GGNet': 'summary_results_ggnet',
    'MOBNet': 'summary_results_mobnet',
    'SHFFNet': 'summary_results_shffnet',
    'EffNet': 'summary_results_effnet',
    'MNASNet': 'summary_results_mnasnet',
    # adicionais (usados se existirem):
    'ResNet': 'summary_results_resnet',
    'DenseNet': 'summary_results_densenet',
    'VGG': 'summary_results_vgg',
}

def resolve_datas_dir() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    datas_dir = os.path.join(root, 'datas')
    os.makedirs(datas_dir, exist_ok=True)
    return datas_dir

def ensure_datas_mirror() -> None:
    """Copia os summary_results das notebooks/ para datas/ por arquitetura."""
    root = os.path.dirname(os.path.abspath(__file__))
    notebooks_dir = os.path.join(root, 'notebooks')
    datas_dir = resolve_datas_dir()
    for sub in set(MODEL_SUMMARY_DIRS.values()):
        src_dir = os.path.join(notebooks_dir, sub)
        dst_dir = os.path.join(datas_dir, sub)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for csv_path in glob.glob(os.path.join(src_dir, '*.csv')):
            try:
                shutil.copy2(csv_path, os.path.join(dst_dir, os.path.basename(csv_path)))
            except Exception:
                pass

def discover_model_csvs(model_name: str) -> List[str]:
    root = os.path.dirname(os.path.abspath(__file__))
    datas_dir = resolve_datas_dir()
    notebooks_dir = os.path.join(root, 'notebooks')
    sub = MODEL_SUMMARY_DIRS.get(model_name)
    paths: List[str] = []
    if sub:
        base = os.path.join(datas_dir, sub)
        paths = sorted(glob.glob(os.path.join(base, '*.csv')))
        if not paths:
            base_nb = os.path.join(notebooks_dir, sub)
            paths = sorted(glob.glob(os.path.join(base_nb, '*.csv')))
    if not paths:
        paths = sorted(glob.glob(os.path.join(datas_dir, '**', f'*{model_name}*fold*results*.csv'), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(notebooks_dir, '**', f'*{model_name}*fold*results*.csv'), recursive=True))
    return paths

def resolve_paths_outputs() -> tuple[str, str]:
    root = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(root, 'outputs')
    tables_dir = os.path.join(outputs_dir, 'tables')
    ensemble_output_base = os.path.join(tables_dir, 'Ensemble_Between_Models')
    os.makedirs(ensemble_output_base, exist_ok=True)
    return tables_dir, ensemble_output_base


# ------------------------------
# Verificação de pré-condições SA antes de MA
# ------------------------------
def expected_sa_paths(tables_dir: str, model: str, ensemble_type: str, weight_metric: str) -> Dict[str, str]:
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
    paths = expected_sa_paths(tables_dir, model, ensemble_type, weight_metric)
    status = {k: os.path.exists(v) for k, v in paths.items()}
    all_ok = all(status.values())
    return all_ok, status


def ensure_sa_outputs(models: List[str], ensemble_type: str, weight_metric: str) -> List[str]:
    tables_dir, _ = resolve_paths_outputs()
    models_ready: List[str] = []
    for m in models:
        ok, status = check_sa_ready(tables_dir, m, ensemble_type, weight_metric)
        if ok:
            models_ready.append(m)
            continue
        # Tentar gerar SA para este modelo
        csvs = discover_model_csvs(m)
        if not csvs:
            print(f"[AVISO] Não encontrei CSVs de folds para o modelo {m}. Pulando este modelo no MA.")
            continue
        print(f"[INFO] Gerando ensembles SA para {m} ({ensemble_type})...")
        per = PerModelEnsembler(model_name=m, ensemble_type=ensemble_type, weight_metric=weight_metric)
        try:
            per.run_tile_level(csvs)
            per.run_image_level(csvs)
            per.run_patient_level(csvs)
        except Exception as e:
            print(f"[ERRO] Falha ao gerar SA para {m}: {e}")
            continue
        # verificar novamente
        ok2, _ = check_sa_ready(tables_dir, m, ensemble_type, weight_metric)
        if ok2:
            models_ready.append(m)
            # exportar para results imediatamente
            export_sa_to_results(tables_dir, m, ensemble_type, weight_metric)
        else:
            print(f"[AVISO] SA incompleto para {m}, não será incluído no MA.")
    return models_ready


# ------------------------------
# Exportação para pasta results/ (SA e MA)
# ------------------------------
def _type_folder_name(ensemble_type: str, weight_metric: str) -> str:
    return f"weighted_{weight_metric}" if ensemble_type == 'weighted' else ensemble_type

def _save_confusion_matrix_png(metrics_json_path: str, labels: List[str], png_out_dir: str) -> None:
    if not _HAS_MPL:
        return
    try:
        with open(metrics_json_path, 'r') as f:
            m = json.load(f)
        cm = m.get('confusion_matrix')
        if not isinstance(cm, list):
            return
        cm_arr = np.array(cm)
        os.makedirs(png_out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm_arr, cmap='Blues')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.colorbar(im, ax=ax)
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j, i, int(cm_arr[i, j]), ha='center', va='center', color='black')
        plt.tight_layout()
        fig.savefig(os.path.join(png_out_dir, 'confusion_matrix.png'))
        plt.close(fig)
    except Exception:
        pass

def _safe_literal_eval(x):
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return {}

def _plot_roc_pr_per_class(csv_path: str, out_dir: str) -> None:
    """Gera gráficos ROC e Precision-Recall por classe a partir de um CSV de ensemble.
    Procura colunas de probabilidades conhecidas e usa true_label como referência.
    """
    if not (_HAS_MPL and _HAS_SKLEARN):
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return

    # Colunas candidatas que podem conter dict de probabilidades por classe
    prob_cols = [
        'final_probs',
        'mean_probs_per_class',
        'mean_probs_per_class_image',
        'mean_probs_per_class_tile',
        'probabilities',
    ]
    probs_col = None
    for c in prob_cols:
        if c in df.columns:
            probs_col = c
            break
    if probs_col is None:
        # Sem probabilidades, não gera curvas
        return

    # Converter eventuais strings em dict
    df[probs_col] = df[probs_col].apply(lambda v: v if isinstance(v, dict) else _safe_literal_eval(v))
    if df[probs_col].isna().all():
        return

    # Descobrir lista de classes
    labels_set = set()
    for d in df[probs_col].dropna():
        if isinstance(d, dict):
            labels_set.update(d.keys())
    # também incluir labels verdadeiros caso não apareçam nas probs
    if 'true_label' in df.columns:
        labels_set.update(df['true_label'].dropna().unique().tolist())
    labels_list = sorted(list(labels_set))
    if not labels_list:
        return

    os.makedirs(out_dir, exist_ok=True)

    # Para cada classe, calcular ROC e PR
    for cls in labels_list:
        try:
            # binarizar y_true
            y_true = (df['true_label'] == cls).astype(int)
            # score da classe
            y_score = df[probs_col].apply(lambda d: float(d.get(cls, 0.0)) if isinstance(d, dict) else 0.0).astype(float)

            # Ignorar se não há positivos ou variação
            if y_true.sum() == 0 or y_true.nunique() < 2:
                continue

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC - Classe {cls}')
            ax.legend(loc='lower right')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f'roc_{cls}.png'))
            plt.close(fig)

            # Precision-Recall
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.plot(recall, precision, color='green', lw=2)
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f'Precision-Recall - Classe {cls}')
            plt.tight_layout()
            fig2.savefig(os.path.join(out_dir, f'pr_{cls}.png'))
            plt.close(fig2)
        except Exception:
            # evitar que falhas em uma classe interrompam o processo
            continue

def export_sa_to_results(tables_dir: str, model: str, ensemble_type: str, weight_metric: str) -> None:
    """Copia artefatos SA para results/SA/<MODEL>/<tipo>/<Level> e gera PNG de matriz de confusão."""
    paths = expected_sa_paths(tables_dir, model, ensemble_type, weight_metric)
    type_folder = _type_folder_name(ensemble_type, weight_metric)
    root = os.path.dirname(os.path.abspath(__file__))
    results_base = os.path.join(root, 'outputs', 'results', 'SA', model, type_folder)

    levels = {
        'Tile': ('tile_csv', 'tile_metrics'),
        'Image': ('image_csv', 'image_metrics'),
        'Patient': ('patient_csv', 'patient_metrics'),
    }
    for level, (csv_key, met_key) in levels.items():
        csv_src = paths[csv_key]
        met_src = paths[met_key]
        if not (os.path.exists(csv_src) and os.path.exists(met_src)):
            continue
        out_dir = os.path.join(results_base, level)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(csv_src, os.path.join(out_dir, os.path.basename(csv_src)))
        shutil.copy2(met_src, os.path.join(out_dir, os.path.basename(met_src)))
        # tentar inferir labels do JSON
        try:
            with open(met_src, 'r') as f:
                m = json.load(f)
            report = m.get('classification_report', {})
            labels = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
            _save_confusion_matrix_png(met_src, labels, out_dir)
        except Exception:
            pass
        # gerar ROC/PR por classe
        try:
            _plot_roc_pr_per_class(csv_src, out_dir)
        except Exception:
            pass

def export_ma_to_results(ensemble_type: str, weight_metric: str, level: str, csv_path: str, metrics_path: str) -> None:
    """Copia artefatos MA para results/MA/<tipo>/<Level> e gera PNG de matriz de confusão."""
    root = os.path.dirname(os.path.abspath(__file__))
    type_folder = _type_folder_name(ensemble_type, weight_metric)
    out_dir = os.path.join(root, 'outputs', 'results', 'MA', type_folder, level)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(csv_path):
        shutil.copy2(csv_path, os.path.join(out_dir, os.path.basename(csv_path)))
    if os.path.exists(metrics_path):
        shutil.copy2(metrics_path, os.path.join(out_dir, os.path.basename(metrics_path)))
        # tentar inferir labels para PNG
        try:
            with open(metrics_path, 'r') as f:
                m = json.load(f)
            report = m.get('classification_report', {})
            labels = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
            _save_confusion_matrix_png(metrics_path, labels, out_dir)
        except Exception:
            pass
    # gerar ROC/PR por classe
    try:
        _plot_roc_pr_per_class(csv_path, out_dir)
    except Exception:
        pass


def run_between_models(models: List[str], ensemble_type: str, weight_metric: str) -> None:
    base_models_parent_directory, ensemble_save_output_base = resolve_paths_outputs()

    # Garante que SA está pronto antes do MA
    models_ready = ensure_sa_outputs(models, ensemble_type, weight_metric)
    if not models_ready:
        print("[AVISO] Nenhum modelo com SA pronto. Abortando MA.")
        return

    ens = BetweenModelsEnsembler(
        base_models_parent_directory=base_models_parent_directory,
        ensemble_save_output_base=ensemble_save_output_base,
        models_to_include=models_ready,
        ensemble_type=ensemble_type,
        weight_metric=weight_metric,
    )

    print(f"\n[Image Level] Running ensemble between models: {ensemble_type} ...")
    img_csv, img_metrics = ens.run_image_level()
    print(f"Saved: {img_csv}\nMetrics: {img_metrics}")
    export_ma_to_results(ensemble_type, weight_metric, 'Image', img_csv, img_metrics)

    print(f"\n[Tile Level] Running ensemble between models: {ensemble_type} ...")
    tile_csv, tile_metrics = ens.run_tile_level()
    print(f"Saved: {tile_csv}\nMetrics: {tile_metrics}")
    export_ma_to_results(ensemble_type, weight_metric, 'Tile', tile_csv, tile_metrics)

    print(f"\n[Patient Level] Running ensemble between models: {ensemble_type} ...")
    pat_csv, pat_metrics = ens.run_patient_level()
    print(f"Saved: {pat_csv}\nMetrics: {pat_metrics}")
    export_ma_to_results(ensemble_type, weight_metric, 'Patient', pat_csv, pat_metrics)


if __name__ == '__main__':
    # Configure aqui os modelos e o método de ensemble.
    MODELS = ['GGNet', 'MOBNet', 'SHFFNet', 'EffNet', 'MNASNet', 'ResNet', 'DenseNet', 'VGG']
    ENSEMBLE_TYPE = os.environ.get('ENSEMBLE_TYPE', 'hard_voting')  # 'hard_voting' | 'soft_voting' | 'weighted'
    WEIGHT_METRIC = os.environ.get('WEIGHT_METRIC', 'f1_macro')     # usado apenas com 'weighted'
    # Exporta SA existentes para results antes de rodar MA
    tables_dir, _ = resolve_paths_outputs()
    ensure_datas_mirror()
    for m in MODELS:
        export_sa_to_results(tables_dir, m, ENSEMBLE_TYPE, WEIGHT_METRIC)
    run_between_models(MODELS, ENSEMBLE_TYPE, WEIGHT_METRIC)

    # Após executar, criar índice consolidado dos artefatos
    try:
        def _generate_results_index():
            root = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(root, 'outputs', 'results')
            index_path = os.path.join(results_dir, 'index.html')
            entries = []
            for dirpath, dirnames, filenames in os.walk(results_dir):
                rel = os.path.relpath(dirpath, results_dir)
                # ignorar diretório raiz como '.'
                if rel == '.':
                    rel = ''
                # ordenar para estabilidade
                dirnames.sort()
                filenames.sort()
                items = []
                for fname in filenames:
                    href = os.path.join(rel, fname) if rel else fname
                    items.append(f'<li><a href="{href}">{href}</a></li>')
                if items:
                    entries.append(f'<h3>{rel or "/"}</h3><ul>' + '\n'.join(items) + '</ul>')
            html = '<!doctype html><html><head><meta charset="utf-8"><title>Resultados - Índice</title></head><body>' \
                   + '<h1>Resultados - Índice consolidado</h1>' + '\n'.join(entries) + '</body></html>'
            os.makedirs(results_dir, exist_ok=True)
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(html)
            return index_path

        _generate_results_index()
    except Exception:
        pass