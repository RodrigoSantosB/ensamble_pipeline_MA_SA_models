"""Utilitários de métricas e gráficos para SA e MA.

Este módulo fornece funções para:
- Detectar colunas de probabilidades em CSVs de resultados (SA/MA);
- Calcular métricas padrão (accuracy, balanced accuracy, f1 macro, recall weighted, ROC AUC OVR);
- Salvar métricas em JSON e gerar gráficos (ROC, matriz de confusão, classification report);
- Varredura de diretórios `outputs/tables` para consolidar métricas e plots em `outputs/results`.
"""
import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# Reutiliza utilitário existente para converter strings em dicts
from .utils import safe_convert_to_dict, log, resolve_paths_outputs
from .flags import GENERATE_LEVELS, ROC_DETAIL



def _detect_prob_column(df: pd.DataFrame, level: str) -> Optional[str]:
    """Detecta a coluna de probabilidades mais apropriada para ROC.

    Considera colunas típicas geradas pelo pipeline nos contextos SA e MA,
    preferindo `final_probs` e variantes de `mean_probs_per_class` por nível.

    Args:
        df (pd.DataFrame): DataFrame carregado do CSV de resultados.
        level (str): Nível do resultado (`tile`, `image` ou `patient`).

    Returns:
        Optional[str]: Nome da coluna de probabilidades se encontrada; caso
            contrário, retorna None.
    """
    candidates = [
        'final_probs',
        f'mean_probs_per_class_{level}',
        'mean_probs_per_class',
        'mean_probs_per_class_tile',
        'mean_probs_per_class_image',
        'mean_probs_per_class_paciente',
    ]
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def compute_metrics_from_csv(csv_path: str, level: str) -> Dict:
    """Computa métricas padrão a partir de um CSV (SA ou MA).

    Lê o CSV, valida colunas essenciais (`true_label`, `predicted_label`),
    detecta colunas de probabilidade quando disponíveis e calcula métricas
    agregadas incluindo relatório de classificação e matriz de confusão.

    Args:
        csv_path (str): Caminho para o arquivo CSV de resultados.
        level (str): Nível associado ao CSV (`tile`, `image` ou `patient`).

    Returns:
        Dict: Dicionário com chaves como `accuracy`, `balanced_accuracy`,
            `f1_macro`, `recall_weighted`, `roc_auc_ovr`,
            `classification_report`, `confusion_matrix`, `classes` e
            `total_<level>s_predicted`. Em caso de falhas, retorna chaves
            com mensagens `warning`.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {
            'warning': f'Falha ao ler CSV: {csv_path}',
        }

    if 'true_label' not in df.columns or 'predicted_label' not in df.columns:
        return {
            'warning': 'CSV sem colunas esperadas: true_label / predicted_label',
        }

    # Limpa nulos nas colunas alvo
    df_clean = df.dropna(subset=['true_label', 'predicted_label']).copy()
    if df_clean.empty:
        return {
            'warning': 'Sem amostras válidas para métricas.',
        }

    y_true = df_clean['true_label']
    y_pred = df_clean['predicted_label']

    # classes envolvidas
    classes = sorted(list(set(y_true.dropna().unique().tolist()) | set(y_pred.dropna().unique().tolist())))
    label_to_int = {l: i for i, l in enumerate(classes)}

    # scores para ROC
    prob_col = _detect_prob_column(df_clean, level)
    if prob_col is not None:
        df_clean[prob_col] = df_clean[prob_col].apply(safe_convert_to_dict)
        y_scores = np.array([[d.get(cls, 0.0) for cls in classes] for d in df_clean[prob_col]])
    else:
        # hard voting → one-hot baseado em predicted_label
        y_scores = np.zeros((len(y_pred), len(classes)))
        for i, pred_label in enumerate(y_pred):
            if pred_label in label_to_int:
                y_scores[i, label_to_int[pred_label]] = 1.0

    y_true_int = np.array([label_to_int.get(lbl, 0) for lbl in y_true])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    recw = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))

    roc = np.nan
    if len(y_scores) > 0 and y_scores.shape[1] > 1 and len(np.unique(y_true_int)) > 1:
        try:
            roc = float(roc_auc_score(y_true_int, y_scores, multi_class='ovr', average='weighted', labels=list(range(len(classes)))))
        except ValueError:
            roc = np.nan

    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    except Exception:
        report = {}
    try:
        cm = confusion_matrix(y_true, y_pred).tolist()
    except Exception:
        cm = []

    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1m,
        'recall_weighted': recw,
        'roc_auc_ovr': roc,
        'classification_report': report,
        'confusion_matrix': cm,
        'classes': classes,
        f'total_{level}s_predicted': int(len(df_clean)),
    }

    return metrics


def save_metrics_from_csv(csv_path: str, metrics_path: str, level: str) -> Dict:
    """Computa e salva métricas em JSON a partir de um CSV.

    Args:
        csv_path (str): Caminho para o arquivo CSV de resultados.
        metrics_path (str): Caminho do arquivo JSON a ser gerado.
        level (str): Nível do resultado (`tile`, `image` ou `patient`).

    Returns:
        Dict: As métricas calculadas, como retornadas por
            `compute_metrics_from_csv`.
    """
    metrics = compute_metrics_from_csv(csv_path, level)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    return metrics


# ==================================== Funções de Plot ==================================== 
def save_confusion_and_report_from_csv(csv_path: str, out_dir: str, level: str, network: str = '') -> None:
    """Gera matriz de confusão e classification report a partir de um CSV.

    Usa as colunas `true_label` e `predicted_label` para construir os
    gráficos e salva as figuras no diretório de saída especificado.

    Args:
        csv_path (str): Caminho para o CSV contendo `true_label` e `predicted_label`.
        out_dir (str): Diretório onde as figuras serão salvas.
        level (str): Nível associado (`tile`, `image` ou `patient`).
        network (str): Rótulo informativo para o título do gráfico (ex.: `SA/<MODEL>` ou `MA`).

    Returns:
        None
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    if 'true_label' not in df.columns or 'predicted_label' not in df.columns:
        return
    y_true = df['true_label'].tolist()
    y_pred = df['predicted_label'].tolist()
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    os.makedirs(out_dir, exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de Confusão - {level.capitalize()} Level for {network}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_model_{level}_level_ensemble.png"))
    plt.close()

    # Classification Report
    try:
        report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        plt.figure(figsize=(8, 0.5 + 0.4 * len(labels)))
        df_report = pd.DataFrame(report_dict).transpose().round(3)
        sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title(f"Classification Report - {level.capitalize()} Level {network}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"classification_report_model_{level}_level_ensemble.png"))
        plt.close()
    except Exception:
        pass


def plot_roc_from_csv(csv_path: str, ensemble_method: str, out_dir: str, level: str, network: str = '', detail: str = 'per_class') -> None:
    """Plota curvas ROC a partir de um CSV de resultados.

    Quando `ensemble_method` é `hard_voting`, deriva probabilidades via
    one-hot de `predicted_label`. Para métodos baseados em probabilidades,
    tenta localizar colunas como `final_probs` ou `mean_probs_per_class`.

    Args:
        csv_path (str): Caminho para o CSV de resultados.
        ensemble_method (str): Método do ensemble (`hard_voting`, `soft_voting`, `weighted`).
        out_dir (str): Diretório onde a figura ROC será salva.
        level (str): Nível associado (`tile`, `image` ou `patient`).
        network (str): Rótulo informativo no título (ex.: `SA/<MODEL>` ou `MA`).
        detail (str): `per_class` para curvas por classe + micro-average;
            `macro_micro` para exibir apenas micro-average.

    Returns:
        None
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return

    y_true = df['true_label'].tolist()
    classes = sorted(list(set(y_true)))

    if ensemble_method == 'hard_voting':
        y_pred = df['predicted_label'].tolist()
        y_probs = (np.eye(len(classes))[np.array([classes.index(p) if p in classes else 0 for p in y_pred])])
    else:
        prob_col_name_candidates = ['final_probs', f'mean_probs_per_class_{level}', 'mean_probs_per_class']
        prob_col_name = None
        for c in prob_col_name_candidates:
            if c in df.columns:
                prob_col_name = c
                break
        if prob_col_name is None:
            return
        df[prob_col_name] = df[prob_col_name].apply(safe_convert_to_dict)
        for d in df[prob_col_name].dropna():
            classes.extend([cls for cls in d.keys() if cls not in classes])
        classes = sorted(list(set(classes)))
        y_probs = np.array([[d.get(cls, 0.0) for cls in classes] for d in df[prob_col_name]])

    from sklearn.preprocessing import label_binarize
    try:
        y_true_bin = label_binarize(y_true, classes=classes)
    except Exception:
        return

    # Micro-average ROC
    try:
        from sklearn.metrics import roc_curve, auc
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
    except Exception:
        return

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_micro, tpr_micro, label=f"Micro-average ROC (AUC = {roc_auc_micro:.2f})", color="deeppink", linestyle=":", linewidth=4)

    if detail == 'per_class':
        from itertools import cycle
        n_classes = len(classes)
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "purple"])
        for i, color in zip(range(n_classes), colors):
            try:
                fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                auc_i = auc(fpr_i, tpr_i)
                plt.plot(fpr_i, tpr_i, color=color, lw=2, label=f"ROC {classes[i]} (AUC = {auc_i:.2f})")
            except Exception:
                continue

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {level.capitalize()} Level for {network} - {ensemble_method}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"roc_curve_model_{level}_level_{ensemble_method}.png"))
    plt.close()


def generate_metrics_plots(csv_path: str, ensemble_method: str, out_dir: str, level: str, network: str = '', detail: str = 'per_class') -> None:
    """Gera métricas e gráficos (ROC, confusão, relatório) a partir de um CSV.

    Calcula métricas agregadas, salva o JSON de métricas e renderiza os
    gráficos de ROC e classificação em `outputs/results/<network>/<level>`.

    Args:
        csv_path (str): Caminho para o CSV de resultados.
        ensemble_method (str): Método do ensemble (`hard_voting`, `soft_voting`, `weighted`).
        out_dir (str): Ignorado na implementação atual; os resultados são salvos
            em um diretório padrão dentro de `outputs/results`.
        level (str): Nível associado (`tile`, `image` ou `patient`).
        network (str): Rótulo informativo para subpasta (ex.: `SA/<MODEL>` ou `MA`).
        detail (str): `per_class` para curvas por classe + micro-average;
            `macro_micro` para apenas micro-average.

    Returns:
        None
    """
    # Define diretório de resultados: outputs/results/<network>/<level>
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        root = os.path.dirname(os.path.abspath(__file__))
    level_folder = level.lower()
    results_dir = os.path.join(root, 'outputs', 'results', network, level_folder)
    os.makedirs(results_dir, exist_ok=True)

    # Determina nome do arquivo de métricas no results
    base_name = f"metrics_{level_folder}_level"
    weight_metric = None
    if ensemble_method == 'weighted':
        try:
            import re
            m = re.search(r"weighted_([A-Za-z0-9_]+)\.csv$", os.path.basename(csv_path))
            if m:
                weight_metric = m.group(1)
        except Exception:
            weight_metric = None
    if ensemble_method == 'weighted' and weight_metric:
        metrics_filename = f"{base_name}_weighted_{weight_metric}.json"
    else:
        metrics_filename = f"{base_name}_{ensemble_method}.json"
    metrics_path = os.path.join(results_dir, metrics_filename)

    # Calcula e salva métricas
    try:
        metrics = compute_metrics_from_csv(csv_path, level_folder)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        log(f"[METRICS] Salvo: {metrics_path}")
    except Exception as e:
        log(f"[AVISO] Falha ao calcular/salvar métricas para {csv_path}: {e}")

    # Gera gráficos (ROC, Confusion Matrix e Classification Report) no diretório de resultados
    try:
        plot_roc_from_csv(csv_path, ensemble_method, results_dir, level=level_folder, network=network, detail=detail)
    except Exception as e:
        log(f"[AVISO] Falha ao gerar ROC para {csv_path}: {e}")
    try:
        save_confusion_and_report_from_csv(csv_path, results_dir, level=level_folder, network=network)
    except Exception as e:
        log(f"[AVISO] Falha ao gerar matriz de confusão/relatório para {csv_path}: {e}")


# ========= Helpers para varrer outputs/tables e gerar métricas/plots ========= #
def _infer_ensemble_from_filename(filename: str) -> tuple[str, Optional[str]]:
    """Infere o método de ensemble e métrica de peso a partir do nome do arquivo.

    Args:
        filename (str): Caminho ou nome do arquivo CSV.

    Returns:
        tuple[str, Optional[str]]: Par contendo o método inferido
            (`hard_voting`, `soft_voting` ou `weighted`) e a métrica de peso
            se aplicável.
    """
    name = os.path.basename(filename)
    if 'weighted_' in name:
        try:
            import re
            m = re.search(r"weighted_([A-Za-z0-9_]+)\.csv$", name)
            return ('weighted', m.group(1) if m else None)
        except Exception:
            return ('weighted', None)
    if 'soft_voting' in name:
        return ('soft_voting', None)
    if 'hard_voting' in name:
        return ('hard_voting', None)
    # fallback: tenta por diretório pai
    parent = os.path.dirname(filename)
    if parent.endswith('_soft_voting'):
        return ('soft_voting', None)
    if parent.endswith('_hard_voting'):
        return ('hard_voting', None)
    if parent.endswith('_weighted'):
        return ('weighted', None)
    return ('hard_voting', None)


def generate_sa_metrics_from_tables(models: Optional[List[str]] = None) -> None:
    """Gera métricas/plots para SA varrendo `outputs/tables/<MODEL>`.

    Percorre subpastas `Ensemble_<level>_level_<type>` por modelo e nível,
    identifica arquivos CSV resultantes e renderiza métricas e gráficos em
    `outputs/results/SA/<MODEL>/<LEVEL>`.

    Args:
        models (Optional[List[str]]): Lista explícita de modelos a processar.
            Se None, descobre automaticamente a partir de `outputs/tables`.

    Returns:
        None
    """
    try:
        tables_dir, _ = resolve_paths_outputs()
    except Exception:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tables_dir = os.path.join(root, 'outputs', 'tables')

    # Descobrir modelos
    if not models:
        try:
            models = [d for d in os.listdir(tables_dir) if os.path.isdir(os.path.join(tables_dir, d)) and d != 'Ensemble_Between_Models']
        except Exception:
            models = []

    levels = [lvl for lvl in GENERATE_LEVELS]

    for model in models:
        base = os.path.join(tables_dir, model)
        for lvl in levels:
            # subpastas esperadas
            sub_candidates = [
                f"Ensemble_{lvl}_level_hard_voting",
                f"Ensemble_{lvl}_level_soft_voting",
                f"Ensemble_{lvl}_level_weighted",
            ]
            for sub in sub_candidates:
                sub_dir = os.path.join(base, sub)
                if not os.path.isdir(sub_dir):
                    continue
                # padrões de nome de arquivo
                if lvl == 'tile':
                    patterns = ['ensemble_per_tile_*.csv']
                elif lvl == 'image':
                    patterns = ['ensemble_per_image_*.csv']
                else:
                    patterns = ['ensemble_per_patient_*.csv']
                for pat in patterns:
                    import glob
                    for csv in glob.glob(os.path.join(sub_dir, pat)):
                        method, _ = _infer_ensemble_from_filename(csv)
                        try:
                            # Salva resultados diretamente em outputs/results/SA/<model>/<level>
                            generate_metrics_plots(csv, method, '', lvl, network=f'SA/{model}', detail=ROC_DETAIL)
                        except Exception as e:
                            log(f"[AVISO] SA {model}/{lvl}: falha ao gerar métricas para {csv}: {e}")


def generate_ma_metrics_from_tables() -> None:
    """Gera métricas/plots para MA varrendo `outputs/tables/Ensemble_Between_Models`.

    Procura subpastas por nível (Tile/Image/Patient) e seus CSVs
    correspondentes, gerando métricas e gráficos em `outputs/results/MA/<LEVEL>`.

    Returns:
        None
    """
    try:
        tables_dir, ensemble_base = resolve_paths_outputs()
    except Exception:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ensemble_base = os.path.join(root, 'outputs', 'tables', 'Ensemble_Between_Models')

    if not os.path.isdir(ensemble_base):
        return

    levels = [lvl for lvl in GENERATE_LEVELS]
    for lvl in levels:
        # Mapear subpastas de MA por nível
        if lvl == 'tile':
            prefix = 'TileLevel_Ensemble_Models_'
            file_pattern = 'ensemble_between_models_tile_level*.csv'
        elif lvl == 'image':
            prefix = 'ImageLevel_Ensemble_Models_'
            file_pattern = 'ensemble_between_models_per_image*.csv'
        else:
            prefix = 'PatientLevel_Ensemble_'
            file_pattern = 'ensemble_between_models_per_patient*.csv'

        # Percorrer subpastas com cada tipo
        try:
            subs = [d for d in os.listdir(ensemble_base) if os.path.isdir(os.path.join(ensemble_base, d)) and d.startswith(prefix)]
        except Exception:
            subs = []
        for sub in subs:
            sub_dir = os.path.join(ensemble_base, sub)
            import glob
            for csv in glob.glob(os.path.join(sub_dir, file_pattern)):
                method, _ = _infer_ensemble_from_filename(csv)
                try:
                    generate_metrics_plots(csv, method, '', lvl, network='MA', detail=ROC_DETAIL)
                except Exception as e:
                    log(f"[AVISO] MA {lvl}: falha ao gerar métricas para {csv}: {e}")