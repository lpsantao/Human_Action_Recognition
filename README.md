# Human_Action_Recognition
LSTM-based human action recognition for two datasets: JHMDB and DHG

Exploration of the impact of different feature types in skeleton action recognition solutions with LSTMs and CNNs. For this, several data preprocessing and feature extraction methods were applied, obtaining multiple features of different natures: time-based, spacebased,
and combined, adapted from consolidated state-of-art works. Includes:
- Optimized LSTM and CNN (DD-Net) models
- Preprocessing scripts of two distinct datasets (JHMDB and DHG)
- Feature Engineering: temporal features, geometric features, and combined features.

- Results:
Acc: 85.2% for DHG-14, 81.3% for DHG-28 and 71.1% for JHMDB

## Dependencies
* [Python 3.6](https://www.python.org/downloads/release/python-36/)
* [Scipy](https://scipy.org/install/)
* [Keras](https://keras.io/getting_started/)
* [Tensorflow](https://www.tensorflow.org/install)

## Source
- [JHMD_data_processing.py](JHMD_data_processing.py) and [SHREC_data_processing.py](SHREC_data_processing.py) - data preprocessing for JHMD and DHG datasets. Outputs data in format to feed the LSTM model
- [JHMD_model.py](JHMD_model.py) and [SHREC_model.py](SHREC_model.py) - CNN (DD-NET) for JHMD and DHG datasets.
- [lstm_jhmd.py](lstm_jhmd.py) - LSTM model for JHMD and DHG datasets.
- [util.py](util.py) and [jhmd_utils.py](jhmd_utils.py) - util functions for feature extraction for DHG and JHMD datasets

