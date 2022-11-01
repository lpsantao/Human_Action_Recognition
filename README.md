# Human_Action_Recognition
LSTM-based human action recognition for two datasets: JHMDB and DHG

Exploration of the impact of different feature types in skeleton action recognition solutions with LSTMs. For this, several data preprocessing and feature extraction methods were applied, obtaining multiple features of different natures: time-based, spacebased,
and combined, adapted from consolidated state-of-art works. Includes:
- Optimized LSTM model with two layers
- Preprocessing scripts of two distinct datasets (JHMDB and DHG)
- Feature Engineering: temporal features, geometric features, and combined features.
Results:
Acc: 85.2% for DHG-14, 81.3% for DHG-28 and 71.1% for JHMDB
