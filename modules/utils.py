"""Utilitários de descoberta, caminhos, logging e agregação (paciente).

Inclui helpers para:
- Conversão segura de strings para dicionários;
- Logging com timestamp;
- Resolução de caminhos em `datas/` e `outputs/`;
- Descoberta dinâmica de modelos e seus CSVs;
- Geração direta de tabela em nível de paciente (SA/MA) com hard/soft/weighted.
"""
import os
import ast
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
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
    """Converte uma string representando dict para um dict real.

    Usa `ast.literal_eval` para parsing seguro. Caso falhe ou o valor não
    seja um mapeamento válido, retorna um dicionário vazio.

    Args:
        value: Valor a ser convertido. Pode ser `dict` ou `str` com a
            representação de um dict.

    Returns:
        dict: Dicionário equivalente ou `{}` em caso de falha.
    """
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return {}



def log(msg):
    """Imprime uma mensagem com timestamp padronizado.

    Args:
        msg (str): Mensagem a ser exibida no log.

    Returns:
        None
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")



def resolve_paths() -> tuple[str, str]:
    """Resolve caminhos base para `tables` e subpasta de MA.

    Returns:
        tuple[str, str]: Par `(tables_dir, ensemble_output_base)` dentro do
            diretório `tables`, garantindo existência de `Ensemble_Between_Models`.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.join(root, 'tables')
    ensemble_output_base = os.path.join(tables_dir, 'Ensemble_Between_Models')
    os.makedirs(ensemble_output_base, exist_ok=True)
    return tables_dir, ensemble_output_base


def resolve_datas_dir() -> str:
    """Retorna o caminho absoluto para o diretório `datas/` na raiz do projeto.

    Assume que `datas` está no mesmo nível do diretório `modules`.

    Returns:
        str: Caminho absoluto de `datas/`.
    """
    # 1. Pega o caminho do diretório onde o script atual está (ex: .../modules/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Sobe um nível para o diretório pai (a raiz do projeto)
    project_root = os.path.dirname(script_dir)
    
    # 3. Agora sim, junta o caminho da raiz do projeto com o nome da pasta 'datas'
    datas_dir = os.path.join(project_root, 'datas')
    
    return datas_dir


def ensure_datas_mirror() -> None:
    """Espelha `notebooks/<summary_results_*>` em `datas/<summary_results_*>`.

    Copia arquivos CSV das subpastas de `notebooks` esperadas para o
    diretório `datas/`, criando subpastas quando necessário.

    Returns:
        None
    """
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




def discover_models_and_paths(models_to_find: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Descobre modelos e seus CSVs na pasta `datas/`.

    Args:
        models_to_find (Optional[List[str]]): Lista opcional de nomes de
            modelos a priorizar. Se None, busca todos os `summary_results_*net`.

    Raises:
        FileNotFoundError: Se `datas/` não for encontrado.

    Returns:
        Dict[str, List[str]]: Mapeamento `model_name -> [csv_paths]`.
    """
    datas_dir = resolve_datas_dir()

    if not os.path.isdir(datas_dir):
        raise FileNotFoundError(
            "A pasta 'datas' não foi encontrada. "
            "Não há dados para rodar o ensemble."
        )

    model_dirs = []
    if models_to_find:
        # Modo 1: Usuário especificou quais modelos procurar
        print(f"[INFO] Buscando modelos especificados: {models_to_find}")
        for model_name in models_to_find:
            # Constrói o nome do diretório esperado a partir do nome do modelo
            # Ex: 'EFFNet' -> 'summary_results_effnet'
            prefix = model_name.lower().replace('net', '')
            dir_name = f"summary_results_{prefix}net"
            full_path = os.path.join(datas_dir, dir_name)
            
            if os.path.isdir(full_path):
                model_dirs.append(full_path)
            else:
                print(f"[AVISO] Diretório para o modelo '{model_name}' não encontrado em: {full_path}")
    else:
        # Modo 2: Comportamento original de autodescoberta
        print("[INFO] Buscando todos os modelos disponíveis ('summary_results_*net')...")
        model_dirs = glob.glob(os.path.join(datas_dir, 'summary_results_*net'))

    models_info = {}
    for dir_path in sorted(model_dirs):
        dir_name = os.path.basename(dir_path)
        try:
            model_prefix = dir_name.replace('summary_results_', '').replace('net', '')
            model_name = model_prefix.upper() + 'Net'
            
            # Garante que apenas modelos da lista sejam incluídos, se a lista foi fornecida
            if models_to_find and model_name not in models_to_find:
                continue

            csv_paths = sorted(glob.glob(os.path.join(dir_path, '*.csv')))
            models_info[model_name] = csv_paths
        except (IndexError, AttributeError):
            continue

    # Imprime o relatório final
    print("\n--- Relatório de Descoberta ---")
    model_names_found = list(models_info.keys())
    
    # Se uma lista foi especificada, use essa ordem. Senão, ordem alfabética.
    report_order = models_to_find if models_to_find else sorted(model_names_found)

    for model_name in report_order:
        if model_name not in models_info:
            continue # O diretório pode não ter sido encontrado
        csv_list = models_info[model_name]
        print(f"{model_name}: {len(csv_list)} CSVs encontrados")
        if not csv_list:
            print(f"Aviso: nenhum CSV encontrado para {model_name}")
        print("------------------------------------------------------------")
    
    print(f"[INFO] Modelos configurados para execução: {sorted(model_names_found)}")
    
    return models_info


def resolve_paths_outputs() -> tuple[str, str]:
    """Resolve caminhos de saída para `outputs/tables` e MA.

    Sobe à raiz do projeto e cria/garante `outputs/tables` e
    `outputs/tables/Ensemble_Between_Models`.

    Returns:
        tuple[str, str]: `(tables_dir, ensemble_output_base)` com caminhos absolutos.
    """
    # 1. Pega o diretório do script atual (ex: .../projeto/modules)
    root = os.path.dirname(os.path.abspath(__file__))

    # 2. Sobe um nível para o diretório pai (a raiz do projeto)
    project_root = os.path.dirname(root)

    # 3. Cria o caminho para 'outputs' a partir da raiz do projeto
    outputs_dir = os.path.join(project_root, 'outputs')
    
    # O resto do código funciona como antes, mas agora a partir do caminho base correto
    tables_dir = os.path.join(outputs_dir, 'tables')
    ensemble_output_base = os.path.join(tables_dir, 'Ensemble_Between_Models')
    
    os.makedirs(ensemble_output_base, exist_ok=True)
    
    return tables_dir, ensemble_output_base


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
    """Gera uma tabela consolidada de nível de paciente (CSV + DataFrame).

    Consolida dados vindos de múltiplas fontes (folds ou modelos), agregando
    de `tile → image → patient` quando necessário. Suporta métodos de ensemble
    `hard_voting`, `soft_voting` e `weighted` e calcula incerteza por paciente.

    Args:
        sa_or_ma (str): Contexto de agregação. `SA` para por modelo, `MA` para
            entre modelos.
        inputs: Especificação das fontes. Para `SA`:
            `{"model_name": str, "csv_paths": List[str]}` ou
            `{"model_name": str, "csv_dir": str}`. Para `MA`:
            `{"models": { model_name: {"csv_paths": [...]} ou {"csv_dir": str} }}`.
        ensemble_type (str): Método de ensemble (`hard_voting`, `soft_voting`,
            `weighted`).
        weight_map (dict | None): Pesos por fonte (fold/modelo). Se ausente,
            assume pesos iguais.
        save_output_dir (str | None): Diretório para salvar o CSV resultante.
            Se None, usa `./outputs/tables/manual_patient_level/`.
        prob_col_candidates (List[str] | None): Lista de possíveis colunas com
            probabilidades por classe. Defaults incluem
            `final_probs`, `mean_probs_per_class`, `mean_probs_per_class_image`,
            `mean_probs_per_class_tile`.

    Returns:
        tuple[str, pd.DataFrame]: Par `(csv_path, df)` com caminho do arquivo
            gerado e o DataFrame consolidado.

    Raises:
        ValueError: Quando os dados mínimos não estão presentes (ex.: `patient_id`
            e `true_label`) ou `sa_or_ma` inválido.
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
