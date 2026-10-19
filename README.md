# Efficient and Robust Spiking Neural Networks for sEMG-Based Muscle Fatigue Detection

This repository contains code for muscle fatigue detection from surface electromyography (sEMG), including feature extraction, MLP and spiking neural network models, weight quantization, noise evaluation, and prediction post-processing. The default input has 132 features.

- `SPE/`: model training and evaluation on the SPE dataset.
- `SUE/`: model training and evaluation on the SUE dataset.
- `preprocess_SPE/`: SPE signal preprocessing, feature extraction, and label conversion.
- `preprocess_SUE/`: SUE data preparation, feature extraction, and classical classifiers.

Licensed under [Apache License 2.0](LICENSE).
