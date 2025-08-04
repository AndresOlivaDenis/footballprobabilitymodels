import pandas as pd
from footballprobabilitymodels.data_transformation.bookmakers_odds_features import bm_probabilities_from_odds
from footballprobabilitymodels.data_transformation.ftr_goals_max_cap import ftr_goals_max_cap
from footballprobabilitymodels.fpm_models.fpm_model import FPMModelOne
from footballprobabilitymodels.fpm_models.fpm_team_models.fpm_team_tf_sequential import FPMTeamTFSequentialModel


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Data
    ds_out_path = '../../../data/football_historical_FTR_data_50K.csv'
    data_df = pd.read_csv(ds_out_path)
    #

    # Model Features
    X = pd.merge(
        bm_probabilities_from_odds(data_df[['FT_1X2_H', 'FT_1X2_D', 'FT_1X2_A']]),
        bm_probabilities_from_odds(data_df[['FT_OU_2.50_under', 'FT_OU_2.50_over']]),
        left_index=True, right_index=True
    )
    # Target variables: Home & Away full time goals
    y_home = ftr_goals_max_cap(data_df['FTHG'])
    y_away = ftr_goals_max_cap(data_df['FTAG'])

    # FPM Model
    fpm_model = FPMModelOne(
        home_team_fpm_model=FPMTeamTFSequentialModel(),
        away_team_fpm_model=FPMTeamTFSequentialModel()
    )

    # Training
    fpm_model.fit(X, y_home, y_away)

    # Making predictions
    match_evaluations = fpm_model.eval_match(
        bm_1x2_odds=[2.32, 3.17, 3.56],
        bm_ou_2_50_odds=[1.54, 2.6]
    )

    # Probability of each match result outcome. Column: Home team goals, rows: Away team goals
    print("Probability of each match result outcome. Column: Home team goals, rows: Away team goals")
    print(match_evaluations.goals_matrix)

    # Home team goals probabilities
    print("\nHome team goals probabilities")
    print(match_evaluations.home_team_goals_probabilities)

    # Away team goals probabilities
    print("\nAway team goals probabilities")
    print(match_evaluations.away_team_goals_probabilities)

    # Home / Away team goals expectations
    print("\nHome / Away team goals expectations")
    print(f"home team goals expectation: {match_evaluations.home_team_goals_expectation} ")
    print(f"away team goals expectation: {match_evaluations.away_team_goals_expectation} ")

    # Saving model
    from joblib import dump, load
    dump(fpm_model, '../FPM_TF_SEQUENTIAL')
