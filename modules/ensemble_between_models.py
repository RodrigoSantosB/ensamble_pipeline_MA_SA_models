"""Ensemble entre diferentes modelos em três níveis: tile, image e patient.

Fornece a classe `BetweenModelsEnsembler` que combina saídas de múltiplos
modelos utilizando hard voting, soft voting ou votação ponderada por métricas.
Mantém a compatibilidade com nomes de arquivos e estruturas dos scripts
originais do projeto (SA/MA).
"""
import os
import os
import ast
import json
from collections import Counter
from typing import Dict, List, Tuple, Any

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


class BetweenModelsEnsembler:
    """
    Orquestra ensemble entre diferentes modelos, preservando a lógica e os nomes
    de arquivos dos scripts originais para os níveis: tile, image e patient.

    - ENSEMBLE_TYPE: 'hard_voting' | 'soft_voting' | 'weighted'
    - WEIGHT_METRIC (apenas para 'weighted'): 'accuracy' | 'f1_macro' | 'recall_weighted' | 'roc_auc_ovr'
    - base_models_parent_directory: diretório base onde ficam os resultados por modelo (ex.: .../tables/{MODEL}/...)
    - ensemble_save_output_base: diretório base onde salvar os resultados entre modelos (ex.: .../tables/Ensemble_Between_Models)
    - models_to_include: lista de modelos a serem incluídos no ensemble
    """

    def __init__(
        self,
        base_models_parent_directory: str,
        ensemble_save_output_base: str,
        models_to_include: List[str],
        ensemble_type: str = "hard_voting",
        weight_metric: str = "f1_macro",
    ) -> None:
        """Inicializa o orquestrador de ensemble entre modelos.

        Args:
            base_models_parent_directory: Diretório base contendo os resultados
                individuais por modelo (ex.: `.../outputs/tables/{MODEL}`).
            ensemble_save_output_base: Diretório onde salvar os artefatos de
                ensemble entre modelos.
            models_to_include: Lista com nomes dos modelos a combinar.
            ensemble_type: Tipo de ensemble ("hard_voting", "soft_voting" ou "weighted").
            weight_metric: Métrica de ponderação quando `ensemble_type` é "weighted"
                ("accuracy", "f1_macro", "recall_weighted", "roc_auc_ovr").

        Returns:
            None. Configura atributos internos e diretórios de saída.
        """
        self.base_models_parent_directory = base_models_parent_directory
        self.ensemble_save_output_base = ensemble_save_output_base
        self.models_to_include = models_to_include
        self.ENSEMBLE_TYPE = ensemble_type
        self.WEIGHT_METRIC = weight_metric

    # ------------------------------
    # Utils
    # ------------------------------
    @staticmethod
    def _safe_literal_eval(x: Any) -> Any:
        """Converte strings literais para objetos Python de forma segura.

        Retorna `np.nan` caso falhe ou quando o valor seja NaN. Se `x` já for
        `dict` ou `list`, retorna o próprio valor.

        Args:
            x: Valor que pode ser string, `dict`, `list` ou outros tipos.

        Returns:
            Objeto convertido (por exemplo, `dict`) ou `np.nan` em caso de erro.
        """
        if pd.isna(x):
            return np.nan
        if isinstance(x, (dict, list)):
            return x
        try:
            return ast.literal_eval(str(x))
        except (ValueError, SyntaxError):
            return np.nan

    @staticmethod
    def _majority_vote(labels: List[str]) -> Any:
        """Retorna o rótulo mais frequente (votação simples).

        Args:
            labels: Lista de rótulos previstos.

        Returns:
            Rótulo com maior frequência ou `np.nan` se a lista estiver vazia.
        """
        return Counter(labels).most_common(1)[0][0] if labels else np.nan

    @staticmethod
    def _normalize_dict_probs(d: Dict[str, float], all_labels: List[str]) -> Dict[str, float]:
        """Normaliza um dicionário de probabilidades para conter todas as classes.

        Garante a presença de todas as classes em `all_labels` e normaliza os
        valores para somarem 1. Caso a soma seja zero, aplica distribuição
        uniforme como fallback.

        Args:
            d: Dicionário com probabilidades por classe.
            all_labels: Lista de classes esperadas.

        Returns:
            Dicionário de probabilidades por classe normalizado.
        """
        vec = {label: float(d.get(label, 0.0)) for label in all_labels}
        total = sum(vec.values())
        if total > 0:
            return {k: v / total for k, v in vec.items()}
        # fallback uniforme
        uniform = 1.0 / max(len(all_labels), 1)
        return {k: uniform for k in all_labels}

    # ------------------------------
    # Voting methods (generic)
    # ------------------------------
    def _soft_voting_probs(self, group: pd.DataFrame, probs_col: str, all_labels: List[str]) -> Dict[str, float]:
        """Agrega probabilidades por média (soft voting) em um grupo.

        Args:
            group: DataFrame agrupado por chave (tile, paciente, etc.).
            probs_col: Nome da coluna que contém o dict de probabilidades.
            all_labels: Lista de classes esperadas.

        Returns:
            Dicionário com probabilidades médias por classe.
        """
        mean_probs: Dict[str, List[float]] = {cls: [] for cls in all_labels}
        for _, row in group.iterrows():
            probs_dict = row[probs_col]
            if isinstance(probs_dict, dict):
                for cls, prob in probs_dict.items():
                    mean_probs[cls].append(float(prob))
        averaged = {cls: (np.mean(vals) if vals else 0.0) for cls, vals in mean_probs.items()}
        return self._normalize_dict_probs(averaged, all_labels)

    def _weighted_voting_probs(self, group: pd.DataFrame, probs_col: str, all_labels: List[str], weights: Dict[str, float], model_col: str) -> Dict[str, float]:
        """Agrega probabilidades por média ponderada (weighted voting).

        Args:
            group: DataFrame agrupado por chave (tile, paciente, etc.).
            probs_col: Coluna com dicionários de probabilidades.
            all_labels: Lista de classes esperadas.
            weights: Pesos por modelo (ou outra chave) para ponderação.
            model_col: Coluna que identifica o modelo de cada linha.

        Returns:
            Dicionário com probabilidades médias ponderadas por classe.
        """
        weighted_sum = {cls: 0.0 for cls in all_labels}
        total_weight = 0.0
        for _, row in group.iterrows():
            model_name = row[model_col]
            w = float(weights.get(model_name, 0.0))
            probs_dict = row[probs_col]
            if isinstance(probs_dict, dict):
                for cls, prob in probs_dict.items():
                    weighted_sum[cls] += float(prob) * w
            total_weight += w
        if total_weight <= 0:
            # fallback: soft voting
            return self._soft_voting_probs(group, probs_col, all_labels)
        averaged = {cls: (val / total_weight) for cls, val in weighted_sum.items()}
        return self._normalize_dict_probs(averaged, all_labels)

    # ------------------------------
    # Tile level
    # ------------------------------
    def run_tile_level(self) -> Tuple[str, str]:
        """Executa o ensemble entre modelos no nível de tile.

        - Carrega CSVs de cada modelo, padroniza colunas e constrói um único
          DataFrame.
        - Aplica o método de votação (`hard`, `soft` ou `weighted`) por `(patient_id, tile_name)`.
        - Salva o CSV consolidado e um JSON com métricas.

        Returns:
            Tupla `(csv_path, metrics_path)` com os artefatos gerados.

        Raises:
            RuntimeError: Se não houver dados válidos de nenhum modelo.
            ValueError: Se `self.ENSEMBLE_TYPE` for desconhecido.
        """
        # Input paths per model
        def get_paths(model_name: str) -> Tuple[str, str]:
            subfolder = f"Ensemble_tile_level_{self.ENSEMBLE_TYPE if self.ENSEMBLE_TYPE != 'weighted' else 'weighted'}"
            if self.ENSEMBLE_TYPE == 'weighted':
                csv_name = f"ensemble_per_tile_weighted_{self.WEIGHT_METRIC}.csv"
                metrics_name = f"global_metrics_tile_level_weighted_{self.WEIGHT_METRIC}.json"
            else:
                csv_name = f"ensemble_per_tile_{self.ENSEMBLE_TYPE}.csv"
                metrics_name = f"global_metrics_tile_level_{self.ENSEMBLE_TYPE}.json"
            csv_path = os.path.join(self.base_models_parent_directory, model_name, subfolder, csv_name)
            metrics_path = os.path.join(self.base_models_parent_directory, model_name, subfolder, metrics_name)
            return csv_path, metrics_path

        out_dir = os.path.join(self.ensemble_save_output_base, f"TileLevel_Ensemble_Models_{self.ENSEMBLE_TYPE}")
        os.makedirs(out_dir, exist_ok=True)

        model_frames: List[pd.DataFrame] = []
        all_labels: set = set()
        model_weights: Dict[str, float] = {}

        for model in self.models_to_include:
            csv_path, metrics_path = get_paths(model)
            if not os.path.exists(csv_path):
                print(f"Aviso: Arquivo de tile não encontrado para {model}: {csv_path}. Pulando.")
                continue
            df = pd.read_csv(csv_path)
            # padroniza estrutura e evita colunas duplicadas de 'predicted_label'
            df['mean_probs_per_class_tile'] = df['mean_probs_per_class_tile'].apply(self._safe_literal_eval)
            pred_col = 'voto_majoritario_simples' if 'voto_majoritario_simples' in df.columns else 'predicted_label'
            df_selected = df[['patient_id', 'tile_name', 'true_label', pred_col, 'mean_probs_per_class_tile']].copy()
            df_selected.rename(columns={
                pred_col: 'predicted_label',
                'mean_probs_per_class_tile': 'probabilities',
            }, inplace=True)
            all_labels.update(df_selected['true_label'].dropna().unique())
            for d in df_selected['probabilities'].dropna():
                if isinstance(d, dict):
                    all_labels.update(list(d.keys()))
            df_selected['model_name'] = model
            df_selected = df_selected.set_index(['patient_id', 'tile_name'])
            model_frames.append(df_selected[['true_label', 'predicted_label', 'probabilities', 'model_name']])

            if self.ENSEMBLE_TYPE == 'weighted':
                if not os.path.exists(metrics_path):
                    print(f"Aviso: Métricas não encontradas para {model}: {metrics_path}")
                else:
                    with open(metrics_path, 'r') as f:
                        m = json.load(f)
                        model_weights[model] = float(m.get(self.WEIGHT_METRIC, 0.0))

        if not model_frames:
            raise RuntimeError("Nenhum dado válido encontrado para qualquer modelo (tile level).")

        all_labels_list = sorted(list(all_labels))
        label_to_int = {l: i for i, l in enumerate(all_labels_list)}

        combined = pd.concat(model_frames, ignore_index=False)
        results: List[Dict[str, Any]] = []
        grouped = combined.groupby(['patient_id', 'tile_name'])

        if self.ENSEMBLE_TYPE == 'hard_voting':
            for (pid, tile), group in grouped:
                true_label = group['true_label'].iloc[0]
                pred = self._majority_vote(group['predicted_label'].tolist())
                results.append({'patient_id': pid, 'tile_name': tile, 'true_label': true_label, 'predicted_label': pred, 'final_probs': {}})
        elif self.ENSEMBLE_TYPE == 'soft_voting':
            for (pid, tile), group in grouped:
                true_label = group['true_label'].iloc[0]
                probs = self._soft_voting_probs(group, 'probabilities', all_labels_list)
                pred = max(probs, key=probs.get) if probs else np.nan
                results.append({'patient_id': pid, 'tile_name': tile, 'true_label': true_label, 'predicted_label': pred, 'final_probs': probs})
        elif self.ENSEMBLE_TYPE == 'weighted':
            # normaliza pesos
            total = sum([w for w in model_weights.values() if pd.notna(w)])
            if total <= 0:
                print("Aviso: pesos inválidos, usando pesos uniformes.")
                model_weights = {m: 1.0 / max(len(self.models_to_include), 1) for m in self.models_to_include}
            else:
                model_weights = {m: (model_weights.get(m, 0.0) / total) for m in self.models_to_include}
            for (pid, tile), group in grouped:
                true_label = group['true_label'].iloc[0]
                probs = self._weighted_voting_probs(group, 'probabilities', all_labels_list, model_weights, 'model_name')
                pred = max(probs, key=probs.get) if probs else np.nan
                results.append({'patient_id': pid, 'tile_name': tile, 'true_label': true_label, 'predicted_label': pred, 'final_probs': probs})
        else:
            raise ValueError(f"Tipo de ensemble '{self.ENSEMBLE_TYPE}' não reconhecido.")

        final_df = pd.DataFrame(results)

        # Save CSV
        if self.ENSEMBLE_TYPE == 'weighted':
            csv_name = f"ensemble_between_models_tile_level_weighted_{self.WEIGHT_METRIC}.csv"
        else:
            csv_name = f"ensemble_between_models_tile_level_{self.ENSEMBLE_TYPE}.csv"
        csv_path = os.path.join(out_dir, csv_name)
        final_df.to_csv(csv_path, index=False)

        # Metrics
        df_clean = final_df.dropna(subset=['true_label', 'predicted_label'])
        metrics_path = os.path.join(out_dir, (
            f"metrics_tile_level_weighted_{self.WEIGHT_METRIC}.json" if self.ENSEMBLE_TYPE == 'weighted' else f"metrics_tile_level_{self.ENSEMBLE_TYPE}.json"
        ))
        self._save_global_metrics(df_clean, all_labels_list, label_to_int, metrics_path, level_key='total_tiles', model_weights=model_weights)

        return csv_path, metrics_path

    # ------------------------------
    # Image level
    # ------------------------------
    def run_image_level(self) -> Tuple[str, str]:
        """Executa o ensemble entre modelos no nível de imagem.

        - Carrega CSVs de imagem de cada modelo e consolida em um único
          DataFrame indexado por `patient_id`.
        - Aplica votação (`hard`, `soft` ou `weighted`) por paciente.
        - Salva CSV consolidado e JSON de métricas.

        Returns:
            Tupla `(csv_path, metrics_path)` com os artefatos de saída.

        Raises:
            ValueError: Se `self.ENSEMBLE_TYPE` for desconhecido.
        """
        def get_paths(model_name: str) -> Tuple[str, str]:
            if self.ENSEMBLE_TYPE == 'weighted':
                subfolder = "Ensemble_image_level_weighted"
                csv_name = f"ensemble_per_image_weighted_{self.WEIGHT_METRIC}.csv"
            else:
                subfolder = f"Ensemble_image_level_{self.ENSEMBLE_TYPE}"
                csv_name = f"ensemble_per_image_{self.ENSEMBLE_TYPE}.csv"
            metrics_name = f"ensemble_global_metrics_image_level_{self.ENSEMBLE_TYPE}.json"
            csv_path = os.path.join(self.base_models_parent_directory, model_name, subfolder, csv_name)
            metrics_path = os.path.join(self.base_models_parent_directory, model_name, subfolder, metrics_name)
            return csv_path, metrics_path

        out_dir = os.path.join(self.ensemble_save_output_base, f"ImageLevel_Ensemble_Models_{self.ENSEMBLE_TYPE}")
        os.makedirs(out_dir, exist_ok=True)

        model_frames: Dict[str, pd.DataFrame] = {}
        all_labels: set = set()
        model_weights: Dict[str, float] = {}

        for model in self.models_to_include:
            csv_path, metrics_path = get_paths(model)
            if not os.path.exists(csv_path):
                print(f"Arquivo não encontrado para {model}, pulando: {csv_path}")
                continue
            if self.ENSEMBLE_TYPE == 'weighted' and not os.path.exists(metrics_path):
                print(f"Arquivo de métricas não encontrado para {model}, pulando: {metrics_path}")
                continue
            df = pd.read_csv(csv_path)
            df['model'] = model
            # atualiza classes
            all_labels.update(df['true_label'].dropna().unique())
            df['mean_probs_per_class_image'] = df['mean_probs_per_class_image'].apply(self._safe_literal_eval)
            for d in df['mean_probs_per_class_image'].dropna():
                if isinstance(d, dict):
                    all_labels.update(list(d.keys()))
            model_frames[model] = df.set_index('patient_id')
            if self.ENSEMBLE_TYPE == 'weighted':
                with open(metrics_path, 'r') as f:
                    m = json.load(f)
                    model_weights[model] = float(m.get(self.WEIGHT_METRIC, 0.0))

        all_labels_list = sorted(list(all_labels))
        label_to_int = {l: i for i, l in enumerate(all_labels_list)}

        combined = pd.DataFrame()
        for _, df in model_frames.items():
            combined = pd.concat([combined, df], axis=0)

        results: List[Dict[str, Any]] = []
        grouped = combined.groupby('patient_id')
        if self.ENSEMBLE_TYPE == 'hard_voting':
            for pid, group in grouped:
                true_label = group['true_label'].iloc[0]
                pred = self._majority_vote(group['predicted_label_ensemble_image'].tolist())
                results.append({'patient_id': pid, 'true_label': true_label, 'predicted_label': pred, 'mean_probs_per_class': np.nan})
        elif self.ENSEMBLE_TYPE == 'soft_voting':
            for pid, group in grouped:
                true_label = group['true_label'].iloc[0]
                probs = self._soft_voting_probs(group, 'mean_probs_per_class_image', all_labels_list)
                pred = max(probs, key=probs.get) if probs else np.nan
                results.append({'patient_id': pid, 'true_label': true_label, 'predicted_label': pred, 'mean_probs_per_class': probs})
        elif self.ENSEMBLE_TYPE == 'weighted':
            total = sum([w for w in model_weights.values() if pd.notna(w)])
            if total <= 0:
                print("Aviso: pesos inválidos, usando uniformes.")
                model_weights = {m: 1.0 / max(len(self.models_to_include), 1) for m in self.models_to_include}
            else:
                model_weights = {m: (model_weights.get(m, 0.0) / total) for m in self.models_to_include}
            for pid, group in grouped:
                true_label = group['true_label'].iloc[0]
                probs = self._weighted_voting_probs(group, 'mean_probs_per_class_image', all_labels_list, model_weights, 'model')
                pred = max(probs, key=probs.get) if probs else np.nan
                results.append({'patient_id': pid, 'true_label': true_label, 'predicted_label': pred, 'mean_probs_per_class': probs})
        else:
            raise ValueError(f"Tipo de ensemble '{self.ENSEMBLE_TYPE}' não reconhecido.")

        final_df = pd.DataFrame(results)

        # Save CSV
        suffix = f"_{self.ENSEMBLE_TYPE}"
        if self.ENSEMBLE_TYPE == 'weighted':
            suffix += f"_{self.WEIGHT_METRIC}"
        csv_path = os.path.join(out_dir, f"ensemble_between_models_per_image{suffix}.csv")
        final_df.to_csv(csv_path, index=False)

        # Metrics file name mirrors original pattern
        json_suffix = f"_{self.ENSEMBLE_TYPE}"
        if self.ENSEMBLE_TYPE == 'weighted':
            json_suffix += f"_{self.WEIGHT_METRIC}"
        metrics_path = os.path.join(out_dir, f"metrics_image_level{json_suffix}.json")

        # Prepare metrics
        df_clean = final_df.dropna(subset=['true_label', 'predicted_label'])
        self._save_global_metrics(df_clean, all_labels_list, label_to_int, metrics_path, level_key='total_images', model_weights=model_weights)

        return csv_path, metrics_path

    # ------------------------------
    # Patient level
    # ------------------------------
    def run_patient_level(self) -> Tuple[str, str]:
        """Executa o ensemble entre modelos no nível de paciente.

        - Carrega CSVs de paciente por modelo, consolidando em um único
          DataFrame.
        - Aplica votação (`hard`, `soft` ou `weighted`) por paciente.
        - Salva CSV consolidado e JSON com métricas.

        Returns:
            Tupla `(csv_path, metrics_path)` com os caminhos de saída.

        Raises:
            RuntimeError: Se não houver dados válidos em nenhum modelo.
            ValueError: Se `self.ENSEMBLE_TYPE` for desconhecido.
        """
        def get_paths(model_name: str) -> Tuple[str, str]:
            subfolder = f"Ensemble_patient_level_{self.ENSEMBLE_TYPE if self.ENSEMBLE_TYPE != 'weighted' else 'weighted'}"
            if self.ENSEMBLE_TYPE == 'weighted':
                csv_name = f"ensemble_per_patient_weighted_{self.WEIGHT_METRIC}.csv"
                metrics_name = f"global_metrics_patient_level_weighted_{self.WEIGHT_METRIC}.json"
            else:
                csv_name = f"ensemble_per_patient_{self.ENSEMBLE_TYPE}.csv"
                metrics_name = f"global_metrics_patient_level_{self.ENSEMBLE_TYPE}.json"
            csv_path = os.path.join(self.base_models_parent_directory, model_name, subfolder, csv_name)
            metrics_path = os.path.join(self.base_models_parent_directory, model_name, subfolder, metrics_name)
            return csv_path, metrics_path

        out_dir = os.path.join(self.ensemble_save_output_base, f"PatientLevel_Ensemble_{self.ENSEMBLE_TYPE}")
        os.makedirs(out_dir, exist_ok=True)

        model_frames: List[pd.DataFrame] = []
        all_labels: set = set()
        model_metrics: Dict[str, Dict[str, float]] = {}

        for model in self.models_to_include:
            csv_path, metrics_path = get_paths(model)
            if not os.path.exists(csv_path):
                print(f"Aviso: Arquivo de ensemble por paciente não encontrado para '{model}'. Pulando.")
                continue
            df = pd.read_csv(csv_path)
            if 'patient_id' in df.columns:
                df = df.set_index('patient_id')
            df['mean_probs_per_class_paciente'] = df['mean_probs_per_class_paciente'].apply(self._safe_literal_eval)
            all_labels.update(df['true_label'].dropna().unique())
            for d in df['mean_probs_per_class_paciente'].dropna():
                if isinstance(d, dict):
                    all_labels.update(list(d.keys()))

            # calcula métricas simples por modelo (para pesos se necessário)
            df_clean = df.dropna(subset=['true_label', 'voto_majoritario_simples_paciente'])
            if not df_clean.empty:
                y_true = df_clean['true_label']
                y_pred = df_clean['voto_majoritario_simples_paciente']
                acc = accuracy_score(y_true, y_pred)
                f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
                recw = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                # ROC scores
                y_scores = []
                for d in df_clean['mean_probs_per_class_paciente'].dropna():
                    vec = [float(d.get(lbl, 0.0)) for lbl in sorted(list(all_labels))]
                    total = sum(vec)
                    if total > 0:
                        vec = [v / total for v in vec]
                    else:
                        vec = [1.0 / max(len(all_labels), 1)] * max(len(all_labels), 1)
                    y_scores.append(vec)
                y_scores = np.array(y_scores)
                label_to_int_tmp = {l: i for i, l in enumerate(sorted(list(all_labels)))}
                y_true_int = np.array([label_to_int_tmp[l] for l in y_true])
                roc = np.nan
                if len(y_scores) > 0 and y_scores.shape[1] > 1 and len(np.unique(y_true_int)) > 1:
                    try:
                        roc = roc_auc_score(y_true_int, y_scores, multi_class='ovr', average='weighted', labels=list(label_to_int_tmp.values()))
                    except ValueError:
                        pass
                model_metrics[model] = {
                    'accuracy': acc,
                    'f1_macro': f1m,
                    'recall_weighted': recw,
                    'roc_auc_ovr': roc,
                    'num_samples': len(df_clean),
                }
            else:
                model_metrics[model] = {
                    'accuracy': 0.0,
                    'f1_macro': 0.0,
                    'recall_weighted': 0.0,
                    'roc_auc_ovr': np.nan,
                    'num_samples': 0,
                }

            df_selected = df[['true_label', 'voto_majoritario_simples_paciente', 'mean_probs_per_class_paciente']].copy()
            df_selected.rename(columns={
                'voto_majoritario_simples_paciente': 'predicted_label',
                'mean_probs_per_class_paciente': 'probabilities',
            }, inplace=True)
            df_selected['model'] = model
            model_frames.append(df_selected)

        if not model_frames:
            raise RuntimeError("Nenhum dado válido encontrado para qualquer modelo (patient level).")

        all_labels_list = sorted(list(all_labels))
        label_to_int = {l: i for i, l in enumerate(all_labels_list)}

        combined = pd.concat(model_frames, ignore_index=False)
        results: List[Dict[str, Any]] = []
        grouped = combined.groupby('patient_id')

        if self.ENSEMBLE_TYPE == 'hard_voting':
            for pid, group in grouped:
                true_label = group['true_label'].iloc[0]
                pred = self._majority_vote(group['predicted_label'].tolist())
                results.append({'patient_id': pid, 'true_label': true_label, 'predicted_label': pred, 'final_probs': {}})
        elif self.ENSEMBLE_TYPE == 'soft_voting':
            for pid, group in grouped:
                true_label = group['true_label'].iloc[0]
                probs = self._soft_voting_probs(group, 'probabilities', all_labels_list)
                pred = max(probs, key=probs.get) if probs else np.nan
                results.append({'patient_id': pid, 'true_label': true_label, 'predicted_label': pred, 'final_probs': probs})
        elif self.ENSEMBLE_TYPE == 'weighted':
            # normaliza pesos com base na métrica escolhida
            metric_values = [model_metrics[m].get(self.WEIGHT_METRIC, 0.0) for m in self.models_to_include if model_metrics.get(m, {}).get('num_samples', 0) > 0 and pd.notna(model_metrics[m].get(self.WEIGHT_METRIC, np.nan))]
            total_metric_sum = sum(metric_values)
            if total_metric_sum > 0:
                weights = {m: model_metrics[m].get(self.WEIGHT_METRIC, 0.0) / total_metric_sum for m in self.models_to_include}
            else:
                print("Aviso: Pesos inválidos para o ensemble ponderado. Usando pesos uniformes.")
                weights = {m: 1.0 / max(len(self.models_to_include), 1) for m in self.models_to_include}
            for pid, group in grouped:
                true_label = group['true_label'].iloc[0]
                probs = self._weighted_voting_probs(group, 'probabilities', all_labels_list, weights, 'model')
                pred = max(probs, key=probs.get) if probs else np.nan
                results.append({'patient_id': pid, 'true_label': true_label, 'predicted_label': pred, 'final_probs': probs})
        else:
            raise ValueError(f"Tipo de ensemble '{self.ENSEMBLE_TYPE}' não reconhecido.")

        final_df = pd.DataFrame(results).set_index('patient_id')

        # Save CSV
        if self.ENSEMBLE_TYPE == 'weighted':
            csv_name = f"ensemble_between_models_per_patient_weighted_{self.WEIGHT_METRIC}.csv"
        else:
            csv_name = f"ensemble_between_models_per_patient_{self.ENSEMBLE_TYPE}.csv"
        csv_path = os.path.join(out_dir, csv_name)
        final_df.to_csv(csv_path, index=True)

        # Metrics
        df_clean = final_df.dropna(subset=['true_label', 'predicted_label'])
        metrics_path = os.path.join(out_dir, (
            f"global_metrics_patient_level_weighted_{self.WEIGHT_METRIC}.json" if self.ENSEMBLE_TYPE == 'weighted' else f"global_metrics_patient_level_{self.ENSEMBLE_TYPE}.json"
        ))
        self._save_global_metrics(df_clean, all_labels_list, label_to_int, metrics_path, level_key='total_patients_predicted')

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
        """Calcula e salva métricas globais (accuracy, F1, recall, ROC-AUC).

        - Constrói `y_scores` a partir de `final_probs` ou `mean_probs_per_class`.
        - Calcula métricas clássicas e grava um JSON em `metrics_path`.

        Args:
            df_clean: DataFrame com `true_label` e `predicted_label` sem NaN.
            all_labels_list: Lista ordenada de classes usadas para compor `y_scores`.
            label_to_int: Mapeamento de rótulos para índices inteiros.
            metrics_path: Caminho do arquivo JSON a ser escrito.
            level_key: Chave para indicar o total de amostras (por exemplo, `total_tiles`).
            model_weights: Pesos usados no ensemble ponderado (opcional).

        Returns:
            None. Cria/atualiza o arquivo JSON com as métricas calculadas.
        """
        if df_clean.empty:
            # still create an empty metrics file to match behavior
            with open(metrics_path, 'w') as f:
                json.dump({level_key: 0, 'warning': 'No valid samples for metrics.'}, f, indent=4)
            return

        y_true = df_clean['true_label']
        y_pred = df_clean['predicted_label']

        # Prepare scores for ROC-AUC
        if 'final_probs' in df_clean.columns and df_clean['final_probs'].notna().any():
            y_scores = [
                [row.get(cls, 0.0) for cls in all_labels_list]
                for row in df_clean['final_probs'].fillna({}).tolist()
            ]
            y_scores = np.array(y_scores)
        elif 'mean_probs_per_class' in df_clean.columns and df_clean['mean_probs_per_class'].notna().any():
            y_scores = [
                [row.get(cls, 0.0) for cls in all_labels_list]
                for row in df_clean['mean_probs_per_class'].fillna({}).tolist()
            ]
            y_scores = np.array(y_scores)
        else:
            # hard voting: one-hot vectors
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
            'ensemble_type': self.ENSEMBLE_TYPE,
            'models_included': self.models_to_include,
            'accuracy': acc,
            'f1_macro': f1m,
            'recall_weighted': recw,
            'roc_auc_ovr': roc,
            'classification_report': report,
            'confusion_matrix': cm,
            level_key: len(df_clean),
        }

        if self.ENSEMBLE_TYPE == 'weighted' and model_weights is not None:
            metrics['weight_metric_used'] = self.WEIGHT_METRIC
            metrics['model_weights'] = model_weights

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)