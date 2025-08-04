import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

from footballprobabilitymodels.data_transformation.bookmakers_odds_features import bm_probabilities_from_odds
from footballprobabilitymodels.data_transformation.ftr_goals_max_cap import ftr_goals_max_cap
from footballprobabilitymodels.fpm_models.fpm_team_models.fpm_team_models import FPMTeamModels
from footballprobabilitymodels.fpm_models.fpm_team_models.fpm_team_tf_sequential import FPMTeamTFSequentialModel
from footballprobabilitymodels.fpm_models.fpm_team_models.fpm_team_xgboost import FPMTeamXGBoostModel
from collections import namedtuple

MatchEvaluations = namedtuple('MatchEvaluations',
                              ['goals_matrix',
                               'home_team_goals_probabilities',
                               'away_team_goals_probabilities',
                               'home_team_goals_expectation',
                               'away_team_goals_expectation'])


class FPMModel:

    def __init__(self, home_team_fpm_model: FPMTeamModels, away_team_fpm_model: FPMTeamModels):
        self.home_team_fpm_model = home_team_fpm_model
        self.away_team_fpm_model = away_team_fpm_model

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        self.home_team_fpm_model.fit(X.copy(), y_home)
        self.away_team_fpm_model.fit(X.copy(), y_away)

    def predict_teams_goals_probabilities(self, X: pd.DataFrame | pd.Series | dict):
        home_team_goals_probs = self.home_team_fpm_model.predict(X)
        away_team_goals_probs = self.away_team_fpm_model.predict(X)
        return home_team_goals_probs, away_team_goals_probs

    def evaluate_match(self, X: pd.Series | dict):
        home_team_goals_probs, away_team_goals_probs = self.predict_teams_goals_probabilities(X)
        home_team_goals_expectation = ((home_team_goals_probs.values)*(home_team_goals_probs.index.values)).sum()
        away_team_goals_expectation = ((away_team_goals_probs.values)*(away_team_goals_probs.index.values)).sum()

        goals_outcomes = list(home_team_goals_probs.index)
        goals_matrix = {
            int(home_goal): {
               int(away_goal): home_team_goals_probs[home_goal] * away_team_goals_probs[away_goal]
                for away_goal in goals_outcomes
            }
            for home_goal in goals_outcomes
        }
        goals_matrix = pd.DataFrame(goals_matrix)
        return MatchEvaluations(
            goals_matrix=goals_matrix,
            home_team_goals_probabilities=home_team_goals_probs,
            away_team_goals_probabilities=away_team_goals_probs,
            home_team_goals_expectation=home_team_goals_expectation,
            away_team_goals_expectation=away_team_goals_expectation
        )


class FPMModelOne(FPMModel):
    def eval_match(self, bm_1x2_odds: list, bm_ou_2_50_odds: list):
        X = pd.concat([
            bm_probabilities_from_odds(pd.Series({'FT_1X2_H': bm_1x2_odds[0], 'FT_1X2_D': bm_1x2_odds[1], 'FT_1X2_A': bm_1x2_odds[2]})),
            bm_probabilities_from_odds(pd.Series({'FT_OU_2.50_under': bm_ou_2_50_odds[0], 'FT_OU_2.50_over': bm_ou_2_50_odds[1]}))
        ])
        return self.evaluate_match(X)
