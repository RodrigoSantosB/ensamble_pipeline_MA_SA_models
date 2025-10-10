"""Flags de controle para geração de métricas e gráficos.
Importe estas flags no notebook/pipeline e no main.py para controle centralizado.
"""

# Quais níveis gerar métricas
GENERATE_LEVELS = ['tile', 'image', 'patient']  # exemplo: ['patient']

# Quais tipos de ensemble gerar gráficos
GENERATE_PLOTS_FOR = ['hard_voting', 'soft_voting']  # exemplo: ['soft_voting']

# Detalhe das curvas ROC/PR: 'per_class' ou 'macro_micro'
ROC_DETAIL = 'per_class'

# Número de workers para paralelização
MAX_WORKERS = 4

# Usar threads (True) ou processos (False) no Windows
USE_THREADS = True