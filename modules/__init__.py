"""
Módulos de pipeline modularizados.
Contém classes para ensembles entre modelos e por modelo (tile, imagem, paciente).
"""

from .ensemble_between_models import BetweenModelsEnsembler
from .per_model_ensembler import PerModelEnsembler