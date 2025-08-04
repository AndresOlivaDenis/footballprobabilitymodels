import pandas as pd


def bm_probabilities_from_odds(
        X: pd.Series | pd.DataFrame,
        add_p_label_to_columns: bool = True
    ):
    if isinstance(X, pd.DataFrame):
        return _bm_probabilities_from_odds_df(X, add_p_label_to_columns)

    if isinstance(X, pd.Series):
        return _bm_probabilities_from_odds_ser(X, add_p_label_to_columns)

    return None


def _bm_probabilities_from_odds_df(X: pd.DataFrame, add_p_label_to_columns: bool =True):
    p = X.copy()
    p = 1.0 / p

    norm_f = p.sum(axis=1, skipna=False)
    for i in range(len(X.columns)):
        p.iloc[:, i] = p.iloc[:, i] / norm_f

    if add_p_label_to_columns:
        p.columns = ["P_" + column for column in p.columns]

    return p


def _bm_probabilities_from_odds_ser(X: pd.Series, add_p_label_to_columns: bool =True):
    p = X.copy()
    p = 1.0 / p

    norm_f = p.sum()
    p = p/norm_f

    if add_p_label_to_columns:
        p.index = ["P_" + column for column in p.index]

    return p
