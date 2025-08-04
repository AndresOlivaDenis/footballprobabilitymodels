from abc import abstractmethod
import pandas as pd


class FPMTeamModels:

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError("Abstract method 'fit' should be implemented.")

    @abstractmethod
    def _predict_proba(self, X: pd.DataFrame):
        raise NotImplementedError("Abstract method '_predict_proba' should be implemented.")

    def _predict_proba_ser(self, X: pd.Series):
        preds_df = self._predict_proba(pd.DataFrame({'0': X}).T)
        return preds_df.iloc[0]

    def predict(self, X: pd.DataFrame | pd.Series | dict):
        if isinstance(X, pd.DataFrame):
            return self._predict_proba(X)

        if isinstance(X, dict):
            return self._predict_proba_ser(pd.Series(X))

        if isinstance(X, pd.Series):
            return self._predict_proba_ser(X)

        raise ValueError("Invalid type for input 'X' ")
