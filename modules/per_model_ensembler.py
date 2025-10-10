import os
import ast
import json
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


class PerModelEnsembler:
    """
    Modulariza os scripts de ensemble por modelo (tile / image / patient), com suporte a:
    - hard_voting
    - soft_voting
    - weighted (ponderado por métrica: accuracy, f1_macro, recall_weighted, roc_auc_ovr)

    Esta classe foca em consolidar a lógica encontrada nos scripts:
      - ensamble_table_per_tile*.py
      - ensamble_table_per_image*.py
      - ensamble_table_per_paciente*.py

    Entradas esperadas por CSV (cada CSV referente a um fold/variação do MESMO modelo):
      - patient_id
      - image_id
      - tile_id
      - true_label
      - predicted_label
      - probabilities (dict ou string representando dict)

    Saídas:
      - CSV com resultado do ensemble por nível
      - JSON com métricas globais

    Observação: Caso os CSVs de entrada possuam colunas com nomes ligeiramente diferentes,
    a função _standardize_columns tenta normalizar para o esquema acima.
    """

    def __init__(
        self,
        model_name: str,
        ensemble_type: str = "hard_voting",
        weight_metric: str = "f1_macro",
        save_output_base: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.ENSEMBLE_TYPE = ensemble_type
        self.WEIGHT_METRIC = weight_metric
        # Onde salvar: tables/<model_name>/ (compatível com BetweenModelsEnsembler e SA existentes)
        root = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(root)
        default_base = os.path.join(project_root, 'outputs', 'tables', model_name)
        self.save_output_base = save_output_base or default_base
        os.makedirs(self.save_output_base, exist_ok=True)

        # Cache interno para evitar recomputar níveis
        self._cache_tile_df: Optional[pd.DataFrame] = None
        self._cache_tile_csv: Optional[str] = None
        self._cache_img_df: Optional[pd.DataFrame] = None
        self._cache_img_csv: Optional[str] = None

    # ------------------------------
    # Helpers
    # ------------------------------
    @staticmethod
    def _safe_literal_eval(x: Any) -> Any:
        if isinstance(x, (dict, list)):
            return x
        if pd.isna(x):
            return {}
        try:
            return ast.literal_eval(str(x))
        except Exception:
            return {}

    @staticmethod
    def _majority_vote(labels: List[str]) -> Any:
        if not labels:
            return np.nan
        c = Counter(labels)
        return c.most_common(1)[0][0]

    @staticmethod
    def _normalize_dict_probs(d: Dict[str, float], all_labels: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {k: float(d.get(k, 0.0)) for k in all_labels}
        s = sum(out.values())
        if s > 0:
            for k in out:
                out[k] = out[k] / s
        return out

    def _soft_voting_probs(self, rows: List[Dict[str, Any]], probs_key: str, all_labels: List[str]) -> Dict[str, float]:
        acc: Dict[str, float] = {k: 0.0 for k in all_labels}
        n = 0
        for row in rows:
            d = self._normalize_dict_probs(self._safe_literal_eval(row.get(probs_key, {})), all_labels)
            for k, v in d.items():
                acc[k] += v
            n += 1
        if n == 0:
            return {k: 0.0 for k in all_labels}
        return {k: acc[k] / n for k in all_labels}

    def _weighted_voting_probs(self, rows: List[Dict[str, Any]], probs_key: str, all_labels: List[str], weights: Dict[str, float], fold_key: str) -> Dict[str, float]:
        acc: Dict[str, float] = {k: 0.0 for k in all_labels}
        for row in rows:
            fold = row.get(fold_key)
            w = float(weights.get(str(fold), 0.0))
            d = self._normalize_dict_probs(self._safe_literal_eval(row.get(probs_key, {})), all_labels)
            for k, v in d.items():
                acc[k] += v * w
        return acc

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        # Garante colunas mínimas e normaliza nomes comuns
        col_map = {
            'Paciente': 'patient_id',
            'Patient': 'patient_id',
            'patient': 'patient_id',
            'Imagem': 'image_id',
            'Image': 'image_id',
            'image': 'image_id',
            'Tile': 'tile_name',
            'tile': 'tile_name',
            'TrueLabel': 'true_label',
            'Label': 'true_label',
            'PredLabel': 'predicted_label',
            'Predicted': 'predicted_label',
            'prob_vector': 'probabilities',
            'probability_vector': 'probabilities',
            'probs': 'probabilities',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        # Se existir image_path, extrair patient_id, image_id e tile_name
        if 'image_path' in df.columns:
            # patient_id (TCGA-XXXXX)
            if 'patient_id' not in df.columns:
                pid = df['image_path'].str.extract(r'(TCGA-[A-Z0-9\-]+)')[0]
                df['patient_id'] = pid
            # image_id (ex.: TCGA-XXXX-...-TS1 do diretório *_files)
            if 'image_id' not in df.columns:
                imgid = df['image_path'].str.extract(r'/(TCGA-[A-Za-z0-9\-]+)_files/')[0]
                # fallback: se não encontrar com padrão TCGA-..., tenta pegar nome do diretório antes de _files
                missing_mask = imgid.isna()
                if missing_mask.any():
                    alt = df.loc[missing_mask, 'image_path'].str.extract(r'/([A-Za-z0-9\-]+)_files/')[0]
                    imgid.loc[missing_mask] = alt
                df['image_id'] = imgid
            # tile_name (último componente do path)
            if 'tile_name' not in df.columns:
                tname = df['image_path'].str.extract(r'/([^/]+\.(?:jpeg|jpg|png))$')[0]
                df['tile_name'] = tname
        # cria colunas ausentes
        for col in ['patient_id', 'image_id', 'tile_name', 'true_label', 'predicted_label', 'probabilities']:
            if col not in df.columns:
                df[col] = np.nan
        return df

    # ------------------------------
    # Carregar CSVs (um por fold) e consolidar
    # ------------------------------
    def _load_folds(self, csv_paths: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        frames: List[pd.DataFrame] = []
        fold_names: List[str] = []
        for p in csv_paths:
            try:
                df = pd.read_csv(p)
            except Exception:
                print(f"Aviso: não foi possível ler CSV: {p}")
                continue
            df = self._standardize_columns(df)
            fold = os.path.splitext(os.path.basename(p))[0]
            df['fold_name'] = fold
            # garante tipo dict nas probabilidades
            df['probabilities'] = df['probabilities'].apply(self._safe_literal_eval)
            frames.append(df)
            fold_names.append(fold)
        if not frames:
            raise RuntimeError("Nenhum CSV válido foi carregado para o ensemble por modelo.")
        combined = pd.concat(frames, ignore_index=True)
        return combined, fold_names

    # ------------------------------
    # Cálculo de pesos por fold
    # ------------------------------
    def _compute_fold_weights(self, combined: pd.DataFrame, fold_names: List[str], all_labels_list: List[str]) -> Dict[str, float]:
        metrics_by_fold: Dict[str, Dict[str, float]] = {}
        for fold in fold_names:
            dff = combined[combined['fold_name'] == fold]
            dff_clean = dff.dropna(subset=['true_label', 'predicted_label'])
            if dff_clean.empty:
                continue
            y_true = dff_clean['true_label']
            y_pred = dff_clean['predicted_label']

            # Para ROC AUC, tenta usar probabilidades
            y_scores = [
                [self._safe_literal_eval(p).get(cls, 0.0) for cls in all_labels_list]
                for p in dff_clean['probabilities'].tolist()
            ]
            y_scores = np.array(y_scores)
            y_true_int = np.array([all_labels_list.index(lbl) if lbl in all_labels_list else 0 for lbl in y_true])

            acc = accuracy_score(y_true, y_pred)
            f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
            recw = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            roc = np.nan
            if len(y_scores) > 0 and y_scores.shape[1] > 1 and len(np.unique(y_true_int)) > 1:
                try:
                    roc = roc_auc_score(y_true_int, y_scores, multi_class='ovr', average='weighted', labels=list(range(len(all_labels_list))))
                except ValueError:
                    pass

            metrics_by_fold[fold] = {
                'accuracy': float(acc),
                'f1_macro': float(f1m),
                'recall_weighted': float(recw),
                'roc_auc_ovr': float(roc) if not pd.isna(roc) else np.nan,
                'num_samples': int(len(dff_clean))
            }

        # Normaliza pesos
        valid = [metrics_by_fold[f].get(self.WEIGHT_METRIC, 0.0) for f in fold_names if metrics_by_fold.get(f, {}).get('num_samples', 0) > 0 and pd.notna(metrics_by_fold.get(f, {}).get(self.WEIGHT_METRIC, np.nan))]
        s = sum(valid)
        if s > 0:
            weights = {f: metrics_by_fold[f].get(self.WEIGHT_METRIC, 0.0) / s for f in fold_names}
        else:
            print("Aviso: não foi possível calcular pesos válidos. Usando pesos uniformes.")
            weights = {f: 1.0 / max(len(fold_names), 1) for f in fold_names}

        self._last_fold_metrics = metrics_by_fold
        self._last_fold_weights = weights
        return weights

    # ------------------------------
    # Tile level
    # ------------------------------
    def run_tile_level(self, csv_paths: List[str]) -> Tuple[str, str]:
        combined, fold_names = self._load_folds(csv_paths)
        # labels presentes
        all_labels = set(combined['true_label'].dropna().unique().tolist()) | set(combined['predicted_label'].dropna().unique().tolist())
        # também considerar chaves de probabilidade
        for d in combined['probabilities'].dropna():
            if isinstance(d, dict):
                all_labels.update(list(d.keys()))
        all_labels_list = sorted(list(all_labels))
        label_to_int = {l: i for i, l in enumerate(all_labels_list)}

        # pesos por fold (se weighted)
        if self.ENSEMBLE_TYPE == 'weighted':
            weights = self._compute_fold_weights(combined, fold_names, all_labels_list)
        else:
            weights = {f: 1.0 for f in fold_names}

        results: List[Dict[str, Any]] = []
        # Alinhar com scripts de referência: agrupar por paciente e tile
        grouped = combined.groupby(['patient_id', 'tile_name'])
        for (pid, tid), group in grouped:
            rows = group.to_dict(orient='records')
            true_label = group['true_label'].iloc[0]
            # Determinar probs agregadas conforme método
            if self.ENSEMBLE_TYPE == 'hard_voting':
                predicted_labels_folds = group['predicted_label'].tolist()
                pred_ensemble = self._majority_vote(predicted_labels_folds)
                final_probs = {}
            elif self.ENSEMBLE_TYPE == 'soft_voting':
                final_probs = self._soft_voting_probs(rows, 'probabilities', all_labels_list)
                pred_ensemble = max(final_probs, key=final_probs.get) if final_probs else np.nan
            elif self.ENSEMBLE_TYPE == 'weighted':
                final_probs = self._weighted_voting_probs(rows, 'probabilities', all_labels_list, weights, 'fold_name')
                pred_ensemble = max(final_probs, key=final_probs.get) if final_probs else np.nan
            else:
                raise ValueError(f"Tipo de ensemble '{self.ENSEMBLE_TYPE}' não reconhecido.")

            # estatísticas auxiliares (médias/STD das probabilidades originais dos folds)
            probs_list = [self._normalize_dict_probs(self._safe_literal_eval(r.get('probabilities', {})), all_labels_list) for r in rows]
            mean_probs = {k: float(np.mean([p[k] for p in probs_list])) if probs_list else 0.0 for k in all_labels_list}
            std_probs = {k: float(np.std([p[k] for p in probs_list])) if probs_list else 0.0 for k in all_labels_list}
            votos = Counter(group['predicted_label'].tolist())

            # probabilidades do vencedor
            predicted_probability_ensemble = float(final_probs.get(pred_ensemble, np.nan)) if isinstance(final_probs, dict) else np.nan
            prob_mean_winner = float(mean_probs.get(pred_ensemble, np.nan)) if mean_probs else np.nan
            prob_std_winner = float(std_probs.get(pred_ensemble, np.nan)) if std_probs else np.nan
            mean_orig_stddev = float(group['probability_std_dev'].mean()) if 'probability_std_dev' in group.columns else np.nan

            results.append({
                'patient_id': pid,
                'tile_name': tid,
                'image_id': group['image_id'].iloc[0] if 'image_id' in group.columns else np.nan,
                'true_label': true_label,
                'true_label_one_hot': group['true_label_one_hot'].iloc[0] if 'true_label_one_hot' in group.columns else np.nan,
                # nomes de saída alinhados com scripts de referência
                'voto_majoritario_simples': votos.most_common(1)[0][0] if votos else np.nan,
                'distribuicao_votos_simples': dict(votos),
                'predicted_label_ensemble': pred_ensemble,
                'predicted_probability_ensemble': predicted_probability_ensemble,
                'mean_probs_per_class_tile': mean_probs,
                'std_probs_per_class': std_probs,
                'predicted_probability_mean_winner': prob_mean_winner,
                'predicted_probability_std_winner': prob_std_winner,
                'mean_probability_std_dev_original': mean_orig_stddev,
                # colunas auxiliares para compatibilidade cruzada
                'predicted_label': pred_ensemble,
                'final_probs': final_probs,
            })

        final_df = pd.DataFrame(results)

        # salvar CSV (compatível com estrutura SA)
        if self.ENSEMBLE_TYPE == 'weighted':
            csv_name = f"ensemble_per_tile_weighted_{self.WEIGHT_METRIC}.csv"
            metrics_name = f"global_metrics_tile_level_weighted_{self.WEIGHT_METRIC}.json"
            subfolder = 'Ensemble_tile_level_weighted'
        else:
            csv_name = f"ensemble_per_tile_{self.ENSEMBLE_TYPE}.csv"
            metrics_name = f"global_metrics_tile_level_{self.ENSEMBLE_TYPE}.json"
            subfolder = f"Ensemble_tile_level_{self.ENSEMBLE_TYPE}"
        out_dir = os.path.join(self.save_output_base, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, csv_name)
        final_df.to_csv(csv_path, index=False)
        # Atualiza cache
        self._cache_tile_df = final_df.copy()
        self._cache_tile_csv = csv_path

        df_clean = final_df.dropna(subset=['true_label', 'predicted_label'])
        metrics_path = os.path.join(out_dir, metrics_name)
        self._save_global_metrics(df_clean, all_labels_list, label_to_int, metrics_path, level_key='total_tiles_predicted', model_weights=(self._last_fold_weights if self.ENSEMBLE_TYPE == 'weighted' else None))

        return csv_path, metrics_path

    # ------------------------------
    # Image level
    # ------------------------------
    def run_image_level(self, csv_paths: List[str]) -> Tuple[str, str]:
        # Usa cache de tiles se disponível; caso contrário, gera uma vez
        if self._cache_tile_df is None:
            tile_csv, _ = self.run_tile_level(csv_paths)
            tile_df = pd.read_csv(tile_csv)
            self._cache_tile_df = tile_df.copy()
            self._cache_tile_csv = tile_csv
        else:
            tile_df = self._cache_tile_df.copy()
        # Descobrir labels
        all_labels = set(tile_df['true_label'].dropna().unique().tolist()) | set(tile_df['predicted_label'].dropna().unique().tolist())
        # também considerar chaves de probabilidade
        for d in tile_df['final_probs'].dropna():
            if isinstance(d, dict):
                all_labels.update(list(d.keys()))
        all_labels_list = sorted(list(all_labels))
        label_to_int = {l: i for i, l in enumerate(all_labels_list)}

        # também carregar dados originais para média de std_dev por imagem/paciente
        combined, _ = self._load_folds(csv_paths)

        # agrega por paciente (alinhado com scripts de referência)
        results: List[Dict[str, Any]] = []
        grouped = tile_df.groupby('patient_id')
        for pid, group in grouped:
            true_label = group['true_label'].iloc[0]
            rows = group.to_dict(orient='records')
            if self.ENSEMBLE_TYPE == 'hard_voting':
                pred_img = self._majority_vote(group['predicted_label'].tolist())
                final_probs = {}
            elif self.ENSEMBLE_TYPE == 'soft_voting':
                final_probs = self._soft_voting_probs(rows, 'final_probs', all_labels_list)
                pred_img = max(final_probs, key=final_probs.get) if final_probs else np.nan
            elif self.ENSEMBLE_TYPE == 'weighted':
                # como os final_probs dos tiles já incorporam pesos dos folds, aqui usamos soft sobre final_probs
                final_probs = self._soft_voting_probs(rows, 'final_probs', all_labels_list)
                pred_img = max(final_probs, key=final_probs.get) if final_probs else np.nan
            else:
                raise ValueError(f"Tipo de ensemble '{self.ENSEMBLE_TYPE}' não reconhecido.")

            votos_tiles = Counter(group['predicted_label'].tolist())

            # estatísticas auxiliares (médias/STD das probs agregadas dos tiles)
            probs_list = [self._normalize_dict_probs(self._safe_literal_eval(r.get('final_probs', {})), all_labels_list) for r in rows]
            mean_probs = {k: float(np.mean([p[k] for p in probs_list])) if probs_list else 0.0 for k in all_labels_list}
            std_probs = {k: float(np.std([p[k] for p in probs_list])) if probs_list else 0.0 for k in all_labels_list}

            predicted_probability_ensemble_image = float(final_probs.get(pred_img, np.nan)) if isinstance(final_probs, dict) else np.nan
            prob_mean_winner_image = float(mean_probs.get(pred_img, np.nan)) if mean_probs else np.nan
            prob_std_winner_image = float(std_probs.get(pred_img, np.nan)) if std_probs else np.nan

            # média de std_dev original dos tiles por paciente
            mean_orig_stddev_tiles = np.nan
            if 'probability_std_dev' in combined.columns:
                try:
                    mean_orig_stddev_tiles = float(combined[combined['patient_id'] == pid]['probability_std_dev'].mean())
                except Exception:
                    mean_orig_stddev_tiles = np.nan

            results.append({
                'patient_id': pid,
                'true_label': true_label,
                'true_label_one_hot': group['true_label_one_hot'].iloc[0] if 'true_label_one_hot' in group.columns else np.nan,
                'voto_majoritario_simples_tiles': votos_tiles.most_common(1)[0][0] if votos_tiles else np.nan,
                'distribuicao_votos_simples_tiles': dict(votos_tiles),
                'predicted_label_ensemble_image': pred_img,
                'predicted_probability_ensemble_image': predicted_probability_ensemble_image,
                'mean_probs_per_class_image': mean_probs,
                'std_probs_per_class_image': std_probs,
                'predicted_probability_mean_winner_image': prob_mean_winner_image,
                'predicted_probability_std_winner_image': prob_std_winner_image,
                'mean_probability_std_dev_original_tiles': mean_orig_stddev_tiles,
                # compatibilidade
                'predicted_label': pred_img,
                'final_probs': final_probs,
            })

        final_df = pd.DataFrame(results)
        # salvar CSV e métricas (compatível com SA)
        if self.ENSEMBLE_TYPE == 'weighted':
            csv_name = f"ensemble_per_image_weighted_{self.WEIGHT_METRIC}.csv"
            metrics_name = f"ensemble_global_metrics_image_level_weighted.json"
            subfolder = 'Ensemble_image_level_weighted'
        else:
            csv_name = f"ensemble_per_image_{self.ENSEMBLE_TYPE}.csv"
            metrics_name = f"ensemble_global_metrics_image_level_{self.ENSEMBLE_TYPE}.json"
            subfolder = f"Ensemble_image_level_{self.ENSEMBLE_TYPE}"
        out_dir = os.path.join(self.save_output_base, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, csv_name)
        final_df.to_csv(csv_path, index=False)
        # Atualiza cache de imagem
        self._cache_img_df = final_df.copy()
        self._cache_img_csv = csv_path

        df_clean = final_df.dropna(subset=['true_label', 'predicted_label'])
        metrics_path = os.path.join(out_dir, metrics_name)
        self._save_global_metrics(df_clean, all_labels_list, label_to_int, metrics_path, level_key='total_images_predicted', model_weights=(self._last_fold_weights if self.ENSEMBLE_TYPE == 'weighted' else None))

        return csv_path, metrics_path

    # ------------------------------
    # Patient level
    # ------------------------------
    def run_patient_level(self, csv_paths: List[str]) -> Tuple[str, str]:
        # Usa cache de imagem se disponível; caso contrário, gera uma vez
        if self._cache_img_df is None:
            img_csv, _ = self.run_image_level(csv_paths)
            img_df = pd.read_csv(img_csv)
            self._cache_img_df = img_df.copy()
            self._cache_img_csv = img_csv
        else:
            img_df = self._cache_img_df.copy()

        all_labels = set(img_df['true_label'].dropna().unique().tolist()) | set(img_df['predicted_label'].dropna().unique().tolist())
        for d in img_df['final_probs'].dropna():
            if isinstance(d, dict):
                all_labels.update(list(d.keys()))
        all_labels_list = sorted(list(all_labels))
        label_to_int = {l: i for i, l in enumerate(all_labels_list)}

        # também carregar dados originais para média de std_dev por paciente
        combined, _ = self._load_folds(csv_paths)

        results: List[Dict[str, Any]] = []
        grouped = img_df.groupby('patient_id')
        for pid, group in grouped:
            true_label = group['true_label'].iloc[0]
            rows = group.to_dict(orient='records')
            if self.ENSEMBLE_TYPE == 'hard_voting':
                pred_patient = self._majority_vote(group['predicted_label'].tolist())
                final_probs = {}
            elif self.ENSEMBLE_TYPE == 'soft_voting':
                final_probs = self._soft_voting_probs(rows, 'final_probs', all_labels_list)
                pred_patient = max(final_probs, key=final_probs.get) if final_probs else np.nan
            elif self.ENSEMBLE_TYPE == 'weighted':
                # final_probs já ponderados nos níveis anteriores → soft aqui
                final_probs = self._soft_voting_probs(rows, 'final_probs', all_labels_list)
                pred_patient = max(final_probs, key=final_probs.get) if final_probs else np.nan
            else:
                raise ValueError(f"Tipo de ensemble '{self.ENSEMBLE_TYPE}' não reconhecido.")

            votos_imgs = Counter(group['predicted_label'].tolist())
            probs_list = [self._normalize_dict_probs(self._safe_literal_eval(r.get('final_probs', {})), all_labels_list) for r in rows]
            mean_probs = {k: float(np.mean([p[k] for p in probs_list])) if probs_list else 0.0 for k in all_labels_list}
            std_probs = {k: float(np.std([p[k] for p in probs_list])) if probs_list else 0.0 for k in all_labels_list}

            predicted_probability_ensemble_paciente = float(final_probs.get(pred_patient, np.nan)) if isinstance(final_probs, dict) else np.nan
            prob_mean_winner_paciente = float(mean_probs.get(pred_patient, np.nan)) if mean_probs else np.nan
            prob_std_winner_paciente = float(std_probs.get(pred_patient, np.nan)) if std_probs else np.nan

            mean_orig_stddev_tiles = np.nan
            if 'probability_std_dev' in combined.columns:
                try:
                    mean_orig_stddev_tiles = float(combined[combined['patient_id'] == pid]['probability_std_dev'].mean())
                except Exception:
                    mean_orig_stddev_tiles = np.nan

            results.append({
                'patient_id': pid,
                'true_label': true_label,
                'voto_majoritario_simples_paciente': votos_imgs.most_common(1)[0][0] if votos_imgs else np.nan,
                'distribuicao_votos_simples_paciente': dict(votos_imgs),
                'predicted_label_ensemble_paciente': pred_patient,
                'predicted_probability_ensemble_paciente': predicted_probability_ensemble_paciente,
                'mean_probs_per_class_paciente': mean_probs,
                'std_probs_per_class_paciente': std_probs,
                'predicted_probability_mean_winner_paciente': prob_mean_winner_paciente,
                'predicted_probability_std_winner_paciente': prob_std_winner_paciente,
                'mean_probability_std_dev_original_tiles': mean_orig_stddev_tiles,
                # compatibilidade
                'predicted_label': pred_patient,
                'final_probs': final_probs,
            })

        final_df = pd.DataFrame(results)

        if self.ENSEMBLE_TYPE == 'weighted':
            csv_name = f"ensemble_per_patient_weighted_{self.WEIGHT_METRIC}.csv"
            metrics_name = f"global_metrics_patient_level_weighted_{self.WEIGHT_METRIC}.json"
            subfolder = 'Ensemble_patient_level_weighted'
        else:
            csv_name = f"ensemble_per_patient_{self.ENSEMBLE_TYPE}.csv"
            metrics_name = f"global_metrics_patient_level_{self.ENSEMBLE_TYPE}.json"
            subfolder = f"Ensemble_patient_level_{self.ENSEMBLE_TYPE}"
        out_dir = os.path.join(self.save_output_base, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, csv_name)
        final_df.to_csv(csv_path, index=False)

        df_clean = final_df.dropna(subset=['true_label', 'predicted_label'])
        metrics_path = os.path.join(out_dir, metrics_name)
        self._save_global_metrics(df_clean, all_labels_list, label_to_int, metrics_path, level_key='total_patients_predicted', model_weights=(self._last_fold_weights if self.ENSEMBLE_TYPE == 'weighted' else None))

        return csv_path, metrics_path

    # ------------------------------
    # Metrics helper
    # ------------------------------
    def _save_global_metrics(
        self,
        df_clean: pd.DataFrame,
        all_labels_list: List[str],
        label_to_int: Dict[str, int],
        metrics_path: str,
        level_key: str,
        model_weights: Dict[str, float] | None = None,
    ) -> None:
        if df_clean.empty:
            with open(metrics_path, 'w') as f:
                json.dump({level_key: 0, 'warning': 'No valid samples for metrics.'}, f, indent=4)
            return

        y_true = df_clean['true_label']
        y_pred = df_clean['predicted_label']

        # preparar scores para ROC-AUC
        prob_dict_cols = [
            'final_probs',
            'mean_probs_per_class_tile',
            'mean_probs_per_class_image',
            'mean_probs_per_class_paciente',
            'mean_probs_per_class',
        ]
        probs_col = None
        for c in prob_dict_cols:
            if c in df_clean.columns and df_clean[c].notna().any():
                probs_col = c
                break
        if probs_col is not None:
            y_scores = [
                [row.get(cls, 0.0) for cls in all_labels_list]
                for row in df_clean[probs_col].fillna({}).tolist()
            ]
            y_scores = np.array(y_scores)
        else:
            y_scores = np.zeros((len(y_pred), len(all_labels_list)))
            for i, pred_label in enumerate(y_pred):
                if pred_label in label_to_int:
                    y_scores[i, label_to_int[pred_label]] = 1.0

        y_true_int = np.array([label_to_int.get(lbl, 0) for lbl in y_true])

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
        recw = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        roc = np.nan
        if len(y_scores) > 0 and y_scores.shape[1] > 1 and len(np.unique(y_true_int)) > 1:
            try:
                roc = roc_auc_score(y_true_int, y_scores, multi_class='ovr', average='weighted', labels=list(range(len(all_labels_list))))
            except ValueError:
                pass

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()

        metrics = {
            'model': self.model_name,
            'ensemble_type': self.ENSEMBLE_TYPE,
            'accuracy': float(acc),
            'f1_macro': float(f1m),
            'recall_weighted': float(recw),
            'roc_auc_ovr': float(roc) if not pd.isna(roc) else np.nan,
            'classification_report': report,
            'confusion_matrix': cm,
            level_key: int(len(df_clean)),
        }

        if self.ENSEMBLE_TYPE == 'weighted' and model_weights is not None:
            metrics['weight_metric_used'] = self.WEIGHT_METRIC
            metrics['fold_weights'] = model_weights
            metrics['fold_metrics'] = getattr(self, '_last_fold_metrics', {})

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)