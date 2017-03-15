# helitrax
Time series classifier in TensorFlow using GridLSTM RNN

Helitrax is a toolkit for multidimensional time series classification (e.g. financial data) using TensorFlow. The primary module (gridlstm.py) can be run from the command line or imported and used as a library. It will import the training/test data, construct an RNN based on the given hyperparameters, train the model, and evaluate the model performance.

The feature_extraction module is currently configured to process financial time series, namely OHLCV price and volume data. It extracts features from the input series, normalizes the series, and drops the original inputs to retain only the derived features. Most of the features are either technical indicators or pattern recognizers from ta-lib.

The orchestrator performs hyperparameter tuning and searches for the best F1 scores on the test set.

Note - associated datasets and Jupyter notebooks will be posted later.
