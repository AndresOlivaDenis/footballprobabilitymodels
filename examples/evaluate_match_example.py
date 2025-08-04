# Estimate teams goals and match results probabilities from Bookmaker odds
from footballprobabilitymodels.saved_models.saved_models_catalog import SavedModelsCatalogE

# SavedModelsCatalogE provides ready to use calibrated models.
fpm_model = SavedModelsCatalogE.TF_SEQUENTIAL.fpm_model

match_evaluations = fpm_model.eval_match(
    bm_1x2_odds=[2.32, 3.17, 3.56],  # Bookmaker 1x2 Odds [Home team winning odds, Draw odds, Away team winning odds]
    bm_ou_2_50_odds=[1.54, 2.6]      # Bookmaker under/over odds
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
print(f"home team goals expectation: {match_evaluations.home_team_goals_expectation:.4f} ")
print(f"away team goals expectation: {match_evaluations.away_team_goals_expectation:.4f} ")
