
import numpy as np
import pandas as pd
import xgboost as xgb
from footballprobabilitymodels.fpm_models.fpm_team_models.fpm_team_models import FPMTeamModels

XGB_PARAMS_DEFAULT = {
    'objective': 'multi:softprob',
    'max_depth': 10,
    'min_child_weight': 10,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'lambda': 10000.0,  # 10000 "looks good"
    'learning_rate': 0.05,
    'seed': 1234,
    'n_jobs': -1
}
XGB_FIT_PARAMS_DEFAULT = {
    'num_boost_round': 2000,
    'early_stopping_rounds': 100,
    'verbose_eval': 100
}


class FPMTeamXGBoostModel(FPMTeamModels):

    def __init__(self, xgb_params=None, xgb_fit_params=None):
        if xgb_params is None:
            xgb_params = XGB_PARAMS_DEFAULT.copy()
        if xgb_fit_params is None:
            xgb_fit_params = XGB_FIT_PARAMS_DEFAULT.copy()
        self.xgb_params = xgb_params
        self.xgb_fit_params = xgb_fit_params
        self.num_class = None
        self.classes_ = None
        self.target_adjustment_min = None
        self.xgb_model = None

    def fit(self, X, y):
        self.classes_ = np.sort(pd.unique(y))
        self.num_class = len(self.classes_)
        min_class = min(y)

        y_adj = y.copy()
        if min_class < 0:
            self.target_adjustment_min = min_class
            y_adj = y_adj - self.target_adjustment_min

        xgb_params = self.xgb_params.copy()
        xgb_params['num_class'] = self.num_class

        x_dMatrix = xgb.DMatrix(X, label=y_adj)

        self.xgb_model = xgb.train(
            xgb_params,
            x_dMatrix,
            **self.xgb_fit_params,
            evals=[(x_dMatrix, 'train')],
        )

    def _predict_proba(self, X):
        x_dMatrix = xgb.DMatrix(X)
        xgb_val_preds = self.xgb_model.predict(x_dMatrix)
        xgb_val_preds = pd.DataFrame(xgb_val_preds, columns=self.classes_)
        return xgb_val_preds
