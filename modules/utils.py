import os
import json
import ast
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    RocCurveDisplay,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)


def safe_convert_to_dict(value):
    """Converte string representando dict para dict real; retorna {} em caso de falha."""
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return {}


def save_confusion_and_report_from_metrics(metrics_json_path: str, out_dir: str, level: str, network: str = '') -> None:
    """Gera PNGs de matriz de confusão e classification report a partir do JSON de métricas.
    Estilo inspirado em all_metrics_generate.py (seaborn heatmap)."""
    try:
        with open(metrics_json_path, 'r') as f:
            m = json.load(f)
    except Exception:
        return

    classes = m.get('classes')
    cm = m.get('confusion_matrix')
    report_dict = m.get('classification_report')
    if not isinstance(classes, list) or not isinstance(cm, list) or not isinstance(report_dict, dict):
        # fallback: inferir labels do relatório
        report = m.get('classification_report', {})
        classes = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
        cm = m.get('confusion_matrix')
        if not isinstance(cm, list) or not classes:
            return

    os.makedirs(out_dir, exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriz de Confusão - {level.capitalize()} Level for {network}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_model_{level}_level_ensemble.png"))
    plt.close()

    # Classification Report
    try:
        plt.figure(figsize=(8, 0.5 + 0.4 * len(classes)))
        df_report = pd.DataFrame(report_dict).transpose().round(3)
        sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title(f"Classification Report - {level.capitalize()} Level {network}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"classification_report_model_{level}_level_ensemble.png"))
        plt.close()
    except Exception:
        pass


def save_confusion_and_report_from_csv(csv_path: str, out_dir: str, level: str, network: str = '') -> None:
    """Gera matriz de confusão e classification report diretamente do CSV.
    Usa colunas 'true_label' e 'predicted_label'."""
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
    """Plota ROC a partir de um CSV seguindo o template de all_metrics_generate.py.
    detail: 'per_class' para one-vs-rest por classe + micro; 'macro_micro' para apenas micro (e por classe compacta)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return

    y_true = df['true_label'].tolist()
    classes = sorted(list(set(y_true)))

    if ensemble_method == 'hard_voting':
        y_pred = df['predicted_label'].tolist()
        y_probs = label_binarize(y_pred, classes=classes)
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

    y_true_bin = label_binarize(y_true, classes=classes)

    # Micro-average ROC
    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
    except Exception:
        return

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_micro, tpr_micro, label=f"Micro-average ROC (AUC = {roc_auc_micro:.2f})", color="deeppink", linestyle=":", linewidth=4)

    if detail == 'per_class':
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


def run_in_parallel(callables_with_args: List[tuple], max_workers: int = 4, use_threads: bool = True):
    """Executa uma lista de tarefas em paralelo. Cada item é (func, args_dict).
    Retorna lista de resultados na mesma ordem. Ideal para Windows (threads por padrão)."""
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


def log(msg):
    """Função simples de log com timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============================
# Geração direta de tabela em nível de paciente (SA/MA)
# ============================
def generate_patient_level_table(
    sa_or_ma: str,
    inputs,
    ensemble_type: str = 'soft_voting',
    weight_map: dict | None = None,
    save_output_dir: str | None = None,
    prob_col_candidates: List[str] | None = None,
):
    """
    Gera diretamente a TABELA de nível de paciente (CSV + DataFrame) a partir de diretórios/paths
    de folds, abstraindo a agregação dos níveis inferiores (tile→imagem→paciente).

    Objetivo: permitir consolidar o nível paciente sem precisar executar o pipeline completo.

    Parâmetros
    - sa_or_ma: 'SA' para ensemble por modelo; 'MA' para ensemble entre modelos.
      • SA: inputs = {'model_name': str, 'csv_paths': List[str]} OU {'model_name': str, 'csv_dir': str}
      • MA: inputs = {'models': { model_name: {'csv_paths': [...]} OU {'csv_dir': str} }}

    - ensemble_type: 'hard_voting' | 'soft_voting' | 'weighted'
    - weight_map: dict opcional com pesos por fonte (fold/modelo). Se não fornecido, assume pesos iguais.
      • Para SA: chave pode ser o nome do arquivo/fold.
      • Para MA: chave pode ser o nome do modelo.
    - save_output_dir: diretório onde salvar o CSV resultante. Se None, cria automaticamente em ./outputs/tables/manual_patient_level/
    - prob_col_candidates: lista de nomes de coluna com probabilidades por classe. Default tenta: ['final_probs','mean_probs_per_class','mean_probs_per_class_image','mean_probs_per_class_tile']

    Retorna
    - (csv_path, df): caminho do CSV gerado e o DataFrame em memória.

    Entradas esperadas (por CSV)
    - Deve conter ao menos: 'patient_id', 'true_label'.
    - Se disponível:
      • 'image_id' (para agregação robusta de tiles→imagem→paciente quando o CSV é por tile)
      • 'tile_id' (opcional)
      • 'predicted_label' (para hard voting quando não houver probabilidades)
      • Coluna de probabilidades (dict por classe) dentre prob_col_candidates

    Estratégia de agregação
    1) Normalização dos dados por fonte (fold/modelo):
       - Carregar todos os CSVs.
       - Identificar se o CSV é por tile (tem image_id e possivelmente tile_id) ou por imagem.
       - Converter a coluna de probabilidades para dict (safe_convert_to_dict) quando existir.

    2) Agregação dentro da fonte (fold/modelo):
       2.1) tile→imagem (se necessário):
            - soft: média dos vetores P_tile.
            - hard: modo dos rótulos dos tiles; vetor por contagem normalizada.
            - weighted: igual aos anteriores, mas aplicando pesos por tile (se fornecido via weight_map_tile, não obrigatório; por padrão iguais).
       2.2) imagem→paciente:
            - soft: média dos vetores P_image.
            - hard: modo dos rótulos das imagens; vetor de contagem normalizada.
            - weighted: aplicar pesos por imagem (se fornecido; por padrão iguais).

    3) Agregação entre fontes:
       - SA: agrega entre folds do MESMO modelo.
       - MA: agrega entre modelos.
       - soft: média (ou média ponderada por weight_map) dos vetores P_patient por fonte.
       - hard: modo dos rótulos por fonte (ou somatório ponderado de votos).

    4) Cálculo de incerteza:
       - U_patient = 1 - max(P_patient) ou entropia normalizada.

    Observações
    - Quando não existe uma coluna de probabilidades, o método soft/weighted tenta cair para hard usando 'predicted_label'.
    - Pesos: se weight_map não for fornecido, assume pesos iguais para todas as fontes.
    - Classes: inferidas a partir de true_label e/ou chaves dos dicts de probabilidade.

    """
    prob_col_candidates = prob_col_candidates or [
        'final_probs', 'mean_probs_per_class', 'mean_probs_per_class_image', 'mean_probs_per_class_tile'
    ]

    # ---------- helpers internos ----------
    def _read_csvs_from_spec(spec) -> List[pd.DataFrame]:
        paths = []
        if isinstance(spec, dict):
            if 'csv_paths' in spec and spec['csv_paths']:
                paths = spec['csv_paths']
            elif 'csv_dir' in spec and os.path.isdir(spec['csv_dir']):
                paths = [os.path.join(spec['csv_dir'], f) for f in os.listdir(spec['csv_dir']) if f.lower().endswith('.csv')]
        dfs = []
        for p in sorted(paths):
            try:
                dfs.append(pd.read_csv(p))
            except Exception:
                continue
        return dfs

    def _extract_prob_col(df: pd.DataFrame) -> str | None:
        for c in prob_col_candidates:
            if c in df.columns:
                return c
        return None

    def _to_probs_dict_series(df: pd.DataFrame, prob_col: str) -> List[dict]:
        s = df[prob_col].apply(safe_convert_to_dict).fillna({})
        return list(s)

    def _labels_from_df(df: pd.DataFrame, prob_dicts: List[dict]) -> List[str]:
        classes = set(df['true_label'].tolist())
        for d in prob_dicts:
            for k in d.keys():
                classes.add(k)
        return sorted(list(classes))

    def _aggregate_group_soft(prob_dicts: List[dict], classes: List[str], weights: List[float] | None = None) -> np.ndarray:
        if not prob_dicts:
            return np.zeros(len(classes), dtype=float)
        weights = weights or [1.0] * len(prob_dicts)
        wsum = sum(weights)
        if wsum == 0:
            weights = [1.0] * len(prob_dicts)
            wsum = len(prob_dicts)
        mat = np.array([[d.get(cls, 0.0) for cls in classes] for d in prob_dicts], dtype=float)
        w = np.array(weights, dtype=float).reshape(-1, 1)
        return (mat * w).sum(axis=0) / wsum

    def _aggregate_group_hard(labels: List[str], classes: List[str], weights: List[float] | None = None) -> np.ndarray:
        if not labels:
            return np.zeros(len(classes), dtype=float)
        weights = weights or [1.0] * len(labels)
        counts = {c: 0.0 for c in classes}
        for y, w in zip(labels, weights):
            counts[y] = counts.get(y, 0.0) + float(w)
        total = sum(counts.values())
        if total == 0:
            return np.zeros(len(classes), dtype=float)
        return np.array([counts[c] / total for c in classes], dtype=float)

    def _uncertainty(p_vec: np.ndarray) -> float:
        if p_vec.size == 0:
            return 1.0
        m = float(p_vec.max())
        return 1.0 - m

    def _vec_to_dict(p_vec: np.ndarray, classes: List[str]) -> dict:
        return {c: float(p) for c, p in zip(classes, p_vec)}

    # ---------- consolidação por fonte (fold/modelo) ----------
    # Retorna: dict[source_id] -> dict[patient_id] = { 'P': np.ndarray, 'y': str, 'true': str }
    per_source_patient = {}

    if sa_or_ma.upper() == 'SA':
        source_id = inputs.get('model_name', 'MODEL')
        dfs = _read_csvs_from_spec(inputs)
        # concatena todos os CSVs do modelo (folds)
        df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if df_all.empty or ('patient_id' not in df_all.columns) or ('true_label' not in df_all.columns):
            raise ValueError('Dados inválidos para SA: é necessário ter patient_id e true_label.')
        prob_col = _extract_prob_col(df_all)
        prob_dicts = _to_probs_dict_series(df_all, prob_col) if prob_col else []
        classes = _labels_from_df(df_all, prob_dicts)

        # Se houver tiles, agregamos primeiro por imagem
        has_image = 'image_id' in df_all.columns
        has_tile = 'tile_id' in df_all.columns
        if has_image and has_tile:
            # tile -> imagem
            img_records = []
            for img_id, g in df_all.groupby('image_id'):
                if prob_col:
                    p_img = _aggregate_group_soft(_to_probs_dict_series(g, prob_col), classes)
                else:
                    labels = g['predicted_label'].tolist() if 'predicted_label' in g.columns else []
                    p_img = _aggregate_group_hard(labels, classes)
                y_img = classes[int(p_img.argmax())] if len(classes) else None
                true_img = g['true_label'].mode().iat[0] if not g['true_label'].empty else None
                pat_id = g['patient_id'].mode().iat[0] if 'patient_id' in g.columns else None
                img_records.append({'image_id': img_id, 'patient_id': pat_id, 'true_label': true_img, 'P': p_img, 'y': y_img})
            # imagem -> paciente
            df_img = pd.DataFrame(img_records)
            patient_map = {}
            for pat_id, g in df_img.groupby('patient_id'):
                P_list = list(g['P'])
                if ensemble_type == 'hard_voting':
                    labels = g['y'].tolist()
                    p_pat = _aggregate_group_hard(labels, classes)
                else:
                    p_pat = _aggregate_group_soft(P_list, classes)
                y_pat = classes[int(p_pat.argmax())] if len(classes) else None
                true_pat = g['true_label'].mode().iat[0] if not g['true_label'].empty else None
                patient_map[pat_id] = {'P': p_pat, 'y': y_pat, 'true': true_pat}
            per_source_patient[source_id] = patient_map
        else:
            # já é por imagem ou por paciente → agregamos diretamente por paciente
            patient_map = {}
            for pat_id, g in df_all.groupby('patient_id'):
                if prob_col and ensemble_type != 'hard_voting':
                    P_list = _to_probs_dict_series(g, prob_col)
                    p_pat = _aggregate_group_soft(P_list, classes)
                else:
                    labels = g['predicted_label'].tolist() if 'predicted_label' in g.columns else []
                    p_pat = _aggregate_group_hard(labels, classes)
                y_pat = classes[int(p_pat.argmax())] if len(classes) else None
                true_pat = g['true_label'].mode().iat[0] if not g['true_label'].empty else None
                patient_map[pat_id] = {'P': p_pat, 'y': y_pat, 'true': true_pat}
            per_source_patient[source_id] = patient_map

    elif sa_or_ma.upper() == 'MA':
        models_spec = inputs.get('models', {})
        if not models_spec:
            raise ValueError('Para MA, forneça inputs={"models": { model_name: {csv_paths: [...]} } }')
        for model_name, spec in models_spec.items():
            dfs = _read_csvs_from_spec(spec)
            df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            if df_all.empty or ('patient_id' not in df_all.columns) or ('true_label' not in df_all.columns):
                # pula modelos sem dados válidos
                continue
            prob_col = _extract_prob_col(df_all)
            prob_dicts = _to_probs_dict_series(df_all, prob_col) if prob_col else []
            classes = _labels_from_df(df_all, prob_dicts)

            has_image = 'image_id' in df_all.columns
            has_tile = 'tile_id' in df_all.columns
            if has_image and has_tile:
                # tile → imagem → paciente
                img_records = []
                for img_id, g in df_all.groupby('image_id'):
                    if prob_col:
                        p_img = _aggregate_group_soft(_to_probs_dict_series(g, prob_col), classes)
                    else:
                        labels = g['predicted_label'].tolist() if 'predicted_label' in g.columns else []
                        p_img = _aggregate_group_hard(labels, classes)
                    y_img = classes[int(p_img.argmax())] if len(classes) else None
                    true_img = g['true_label'].mode().iat[0] if not g['true_label'].empty else None
                    pat_id = g['patient_id'].mode().iat[0] if 'patient_id' in g.columns else None
                    img_records.append({'image_id': img_id, 'patient_id': pat_id, 'true_label': true_img, 'P': p_img, 'y': y_img})
                df_img = pd.DataFrame(img_records)
                patient_map = {}
                for pat_id, g in df_img.groupby('patient_id'):
                    P_list = list(g['P'])
                    if ensemble_type == 'hard_voting':
                        labels = g['y'].tolist()
                        p_pat = _aggregate_group_hard(labels, classes)
                    else:
                        p_pat = _aggregate_group_soft(P_list, classes)
                    y_pat = classes[int(p_pat.argmax())] if len(classes) else None
                    true_pat = g['true_label'].mode().iat[0] if not g['true_label'].empty else None
                    patient_map[pat_id] = {'P': p_pat, 'y': y_pat, 'true': true_pat}
                per_source_patient[model_name] = patient_map
            else:
                # imagem → paciente direto
                patient_map = {}
                for pat_id, g in df_all.groupby('patient_id'):
                    if prob_col and ensemble_type != 'hard_voting':
                        P_list = _to_probs_dict_series(g, prob_col)
                        p_pat = _aggregate_group_soft(P_list, classes)
                    else:
                        labels = g['predicted_label'].tolist() if 'predicted_label' in g.columns else []
                        p_pat = _aggregate_group_hard(labels, classes)
                    y_pat = classes[int(p_pat.argmax())] if len(classes) else None
                    true_pat = g['true_label'].mode().iat[0] if not g['true_label'].empty else None
                    patient_map[pat_id] = {'P': p_pat, 'y': y_pat, 'true': true_pat}
                per_source_patient[model_name] = patient_map
    else:
        raise ValueError("sa_or_ma deve ser 'SA' ou 'MA'")

    # ---------- agregação entre fontes (folds/modelos) ----------
    # Conjunto de todos os pacientes e classes
    all_patients = set()
    for src_map in per_source_patient.values():
        all_patients.update(src_map.keys())
    # inferir classes pela primeira fonte disponível
    classes_ref = None
    for src_map in per_source_patient.values():
        for v in src_map.values():
            classes_ref = list(v['P'])
            break
        if classes_ref is not None:
            break
    # classes_ref aqui é um vetor de probabilidades; não temos nomes de classes originais.
    # Para manter compatibilidade com o pipeline existente, derivaremos classes a partir de true_label agregados.
    # Coleta de labels verdadeiros
    true_labels_set = set()
    for src_map in per_source_patient.values():
        for v in src_map.values():
            if v['true'] is not None:
                true_labels_set.add(v['true'])
    classes_names = sorted(list(true_labels_set)) if true_labels_set else [str(i) for i in range(len(classes_ref or []))]

    def _argmax_label(p_vec: np.ndarray) -> str:
        idx = int(np.argmax(p_vec)) if p_vec.size else 0
        return classes_names[idx] if idx < len(classes_names) else str(idx)

    # pesos por fonte
    src_names = list(per_source_patient.keys())
    if weight_map:
        weights = [float(weight_map.get(s, 1.0)) for s in src_names]
    else:
        weights = [1.0] * len(src_names)
    wsum_sources = sum(weights) if weights else 1.0
    if wsum_sources == 0:
        weights = [1.0] * len(src_names)
        wsum_sources = len(src_names)

    rows = []
    for pat_id in sorted(all_patients):
        # Coletar P e y por fonte
        P_list = []
        y_list = []
        true_list = []
        active_weights = []
        for s_name, s_map in per_source_patient.items():
            if pat_id in s_map:
                P_list.append(s_map[pat_id]['P'])
                y_list.append(s_map[pat_id]['y'])
                if s_map[pat_id]['true'] is not None:
                    true_list.append(s_map[pat_id]['true'])
                active_weights.append(float(weight_map.get(s_name, 1.0)) if weight_map else 1.0)

        if not P_list and not y_list:
            continue

        # Ensemble entre fontes
        if ensemble_type == 'hard_voting':
            p_pat = _aggregate_group_hard(y_list, classes_names, active_weights)
        elif ensemble_type == 'weighted':
            p_pat = _aggregate_group_soft(P_list, classes_names, active_weights)
        else:  # 'soft_voting'
            p_pat = _aggregate_group_soft(P_list, classes_names)

        y_pat = _argmax_label(p_pat)
        true_pat = pd.Series(true_list).mode().iat[0] if true_list else None
        unc = _uncertainty(p_pat)
        rows.append({
            'patient_id': pat_id,
            'true_label': true_pat,
            'predicted_label': y_pat,
            'final_probs': _vec_to_dict(p_pat, classes_names),
            'uncertainty': float(unc),
            'sources_count': len(P_list) or len(y_list),
        })

    df_pat = pd.DataFrame(rows)
    # salvar CSV
    base_dir = save_output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'tables', 'manual_patient_level')
    os.makedirs(base_dir, exist_ok=True)
    fname = f"manual_patient_level_{sa_or_ma.lower()}_{ensemble_type}.csv"
    csv_path = os.path.join(base_dir, fname)
    try:
        df_pat.to_csv(csv_path, index=False)
    except Exception:
        pass
    return csv_path, df_pat
