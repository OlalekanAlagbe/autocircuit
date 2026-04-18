"""
Pipeline Constants
Shared configuration values used across multiple pipeline scripts.
Centralizing these prevents threshold mismatches between stages.
"""

# Bottleneck convergence threshold: a feature appearing in this fraction
# of critical paths (or more) is considered an information bottleneck.
# Used by Stage 1.5 (cross-circuit analysis) and Stage 2 (visualizations).
BOTTLENECK_CONVERGENCE_THRESHOLD = 0.6

# Feature ID mapping for Neuronpedia API
# GEMMA SAE has 16384 features per layer (16k dictionary)
GEMMA_SAE_DICT_SIZE = 16384

# Model layer counts
GEMMA_TOTAL_LAYERS = 26
QWEN_TOTAL_LAYERS = 36

# Layer group definitions for analysis
# These define processing stages for each model architecture
GEMMA_LAYER_GROUPS = {
    'input': (0, 5),
    'early_proc': (6, 10),
    'middle_proc': (11, 15),
    'late_proc': (16, 20),
    'output': (21, 25),
}

QWEN_LAYER_GROUPS = {
    'input': (0, 5),
    'early_proc': (6, 11),
    'middle_proc': (12, 20),
    'late_proc': (21, 28),
    'output': (29, 35),
}
