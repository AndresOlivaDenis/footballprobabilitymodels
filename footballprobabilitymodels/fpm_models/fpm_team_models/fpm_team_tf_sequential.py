import numpy as np
import pandas as pd

import tensorflow as tf
from footballprobabilitymodels.fpm_models.fpm_team_models.fpm_team_models import FPMTeamModels

TF_SEQUENTIAL_PARAMS_DEFAULT = {
    'hidden_layers': [
        {'units': 5, 'activation': None},
        {'units': 5, 'activation': None},
    ],
    # 'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
    'epochs': 10,
    # 'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
}


class FPMTeamTFSequentialModel(FPMTeamModels):

    def __init__(self, tf_sequential_params: dict | None = None):
        if tf_sequential_params is None:
            tf_sequential_params = TF_SEQUENTIAL_PARAMS_DEFAULT.copy()

        self.tf_sequential_params = tf_sequential_params
        self.num_class = None
        self.classes_ = None
        self.tf_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes_ = np.sort(pd.unique(y))
        self.num_class = len(self.classes_)

        self.tf_model = tf.keras.Sequential()
        for hidden_layer in self.tf_sequential_params['hidden_layers']:
            self.tf_model.add(tf.keras.layers.Dense(units=hidden_layer['units'], activation=hidden_layer['activation']))

        self.tf_model.add(tf.keras.layers.Dense(units=self.num_class, activation='softmax'))

        self.tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.tf_model.fit(
            X, y,
            epochs=self.tf_sequential_params['epochs'],
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
        )

    def _predict_proba(self, X: pd.DataFrame):
        preds = self.tf_model.predict(X)
        preds = pd.DataFrame(preds, columns=self.classes_)
        return preds
