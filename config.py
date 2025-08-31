#--== Dataset ==--
DATASET = 'english'
# DATASET = 'symbolic'
# DATASET = 'minimal'

#--== Perceptual Loss ==--
FEATURE_LAYERS = [2, 7, 12, 14, 16, 21]
LAYER_WEIGHTS = [1.9, 1.0, 0.5, 0.35, 0.25, 0.15]

#--== Training ==--
LEARNING_RATE = 0.00005
TOTAL_EPOCHS = 100