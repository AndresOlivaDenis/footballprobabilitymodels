import pandas as pd


def ftr_goals_max_cap(
        y: pd.Series | pd.Series,
        cap_value: int = 6
    ):
    y_processed = y
    y_processed = y_processed.where(
        (y_processed < cap_value) | y_processed.isna(),
        cap_value
    )
    return y_processed
