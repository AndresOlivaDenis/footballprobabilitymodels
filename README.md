# footballprobabilitymodels
**footballprobabilitymodels** is a Python library for estimating football match outcome probabilities and expected goal distributions using calibrated machine learning models and bookmaker odds.

## What it does
Given:
- **1X2 odds** from bookmakers (win/draw/loss)
- **Over/under 2.5 goals odds**

The model computes:

 Expected number of goals per team
- A full **probability matrix** of possible match scorelines (home vs away goals)
- **Marginal goal distributions** for each team
- **Expected number of goals** per team

This allows advanced evaluation of football matches beyond win/draw/loss.


# Repository Structure

```bash
footballprobabilitymodels/
├── data/                
├── examples/
│   ├── calibrate_model_example.py
│   └── evaluate_match_example.py
├── footballprobabilitymodels/
│   ├── data_transformation
│   ├── fpm_models
│   │   └── fpm_team_models
│   └── saved_models
│   │   ├── calibration_scripts
│   │   └── saved_models_catalog.py
├── README.md
└── requirements.txt
```

# Getting Started

### Prerequisites
- Python 3.10
- tensorflow, xgboost, scikit-learn, numpy, pamdas, joblib

### Install dependencies:
```bash
pip install -r requirements.txt
```

# Example

```python
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
print(f"home team goals expectation: {match_evaluations.home_team_goals_expectation} ")
print(f"away team goals expectation: {match_evaluations.away_team_goals_expectation} ")
```

```
OUT:

Probability of each match result outcome. Column: Home team goals, rows: Away team goals
          0         1         2         3         4         5         6
0  0.125790  0.153491  0.079798  0.028376  0.009294  0.002482  0.000662
1  0.115708  0.141189  0.073402  0.026101  0.008549  0.002283  0.000609
2  0.050706  0.061873  0.032167  0.011438  0.003746  0.001000  0.000267
3  0.016963  0.020699  0.010761  0.003827  0.001253  0.000335  0.000089
4  0.004292  0.005237  0.002723  0.000968  0.000317  0.000085  0.000023
5  0.000871  0.001062  0.000552  0.000196  0.000064  0.000017  0.000005
6  0.000229  0.000280  0.000145  0.000052  0.000017  0.000005  0.000001

Home team goals probabilities
0.0    0.314559
1.0    0.383831
2.0    0.199549
3.0    0.070958
4.0    0.023240
5.0    0.006205
6.0    0.001656

Away team goals probabilities
0.0    0.399892
1.0    0.367841
2.0    0.161198
3.0    0.053928
4.0    0.013645
5.0    0.002768
6.0    0.000729

Home / Away team goals expectations
home team goals expectation: 1.1297 
away team goals expectation: 0.9248 
```
