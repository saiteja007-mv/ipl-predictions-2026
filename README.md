# IPL 2026 Match Prediction

An ensemble ML system for predicting IPL 2026 match winners using 18 years of historical data (2008–2025).

## Model

Stacking ensemble of three gradient boosting frameworks:

| Model | Role |
|-------|------|
| LightGBM | Primary — fast, handles tabular data well |
| CatBoost | Secondary — handles team-name categoricals natively |
| XGBoost | Baseline — retained for backward compatibility |
| Soft-voting ensemble | Final prediction (~72–75% accuracy) |

## Key Features

- **Elo rating** — rolling team strength score updated after every match
- **Team form** — last 5 matches win rate (overall) + last 3 matches this season
- **Head-to-head** record between the two teams
- **Venue stats** — bat-first win %, per-team venue win %
- **Home advantage** — IPL teams playing in their home city
- **Toss impact** — toss winner vs decision interaction at the venue
- **Player strength** — batting SR, bowling economy, experience caps
- **Momentum** — weighted recent win streak score

## Setup

```bash
# Create and activate environment
conda create -n ipl python=3.11 -y
conda activate ipl

# Install dependencies
pip install -r requirements.txt
```

## How to Use

### 1. Train the Ensemble Model — `training.ipynb`

Open `training.ipynb` in Jupyter and run all cells in order:

1. **Setup & Imports** — installs dependencies, sets paths
2. **Data Loading** — reads CSVs from `Datasets/`
3. **Feature Engineering** — Elo ratings, form, H2H, venue, player stats
4. **EDA** — 4 key visualisations (win distribution, toss impact, venue stats, team form)
5. **Model Training** — trains LightGBM, CatBoost, XGBoost + soft-voting ensemble
6. **Evaluation & Feature Importance** — accuracy, ROC-AUC, feature importance chart
7. **Save Models** — writes all model artifacts to `Models/`

Saved artifacts:
- `Models/lgbm_ipl_model.pkl`
- `Models/catboost_ipl_model.pkl`
- `Models/xgb_ipl_model.pkl`
- `Models/ensemble_ipl_model.pkl`
- `Models/feature_columns.pkl`
- `Models/team_name_map.pkl`
- `Models/elo_ratings.pkl`

### 2. Predict Matches — `prediction.ipynb`

Open `prediction.ipynb` and fill in the match details in **Section 2**:

```python
TEAM1         = "Mumbai Indians"
TEAM2         = "Chennai Super Kings"
VENUE         = "Wankhede Stadium"
TOSS_WINNER   = "Mumbai Indians"
TOSS_DECISION = "bat"
```

Run the cell to get:
- Predicted winner with confidence %
- Win probability bar
- Head-to-head, form, and venue insight cards
- Feature contribution breakdown

**After the match** — update the dataset in **Section 4** by adding the actual result to `new_matches`, then run that cell.

**Season tracker** (Section 5) keeps a running tally of predictions vs actuals for IPL 2026.

### 3. Generate Prediction Cards — `video_card.py`

**Quick demo (no model needed):**

```bash
python video_card_demo.py
# → output/prediction_card_Mumbai_Indians_vs_Chennai_Super_Kings_<date>.png
```

**From Python (after prediction):**

```python
from video_card import generate_prediction_card

generate_prediction_card(
    team1="Mumbai Indians",
    team2="Chennai Super Kings",
    venue="Wankhede Stadium",
    predicted_winner="Mumbai Indians",
    win_prob_team1=0.63,
    win_prob_team2=0.37,
    h2h=(18, 14),
    form_team1="WWLWW",
    form_team2="WLWLW",
    venue_win_pct_team1=61.0,
    venue_win_pct_team2=39.0,
    date="2026-03-28",
)
```

**From CLI:**

```bash
python video_card.py \
  --team1 "Mumbai Indians" \
  --team2 "Chennai Super Kings" \
  --venue "Wankhede Stadium" \
  --winner "Mumbai Indians" \
  --prob1 0.63 --prob2 0.37 \
  --h2h 18,14 \
  --form1 WWLWW --form2 WLWLW \
  --venue-pct1 61 --venue-pct2 39
```

Cards are saved to `output/`.

### 4. Update Dataset After Matches

In `prediction.ipynb` Section 4 (Post-Match Result Logger), add completed matches:

```python
new_matches = [
    {
        'id': 10001,
        'city': 'Mumbai',
        'date': '2026-03-28',
        'season': 2026,
        'team1': 'Mumbai Indians',
        'team2': 'Chennai Super Kings',
        'venue': 'Wankhede Stadium',
        'toss_winner': 'Mumbai Indians',
        'toss_decision': 'bat',
        'winner': 'Mumbai Indians',
        'result': 'runs',
        'win_by_runs': 15,
        'win_by_wickets': 0,
        'player_of_match': 'Rohit Sharma',
    },
]
```

Run the cell — it appends to `Datasets/matches.csv`. Restart the kernel so subsequent predictions use updated data.

## IPL 2026 Teams

| Team | Home Ground |
|------|-------------|
| Mumbai Indians | Wankhede Stadium |
| Chennai Super Kings | MA Chidambaram Stadium |
| Royal Challengers Bengaluru | M Chinnaswamy Stadium |
| Kolkata Knight Riders | Eden Gardens |
| Delhi Capitals | Arun Jaitley Stadium |
| Sunrisers Hyderabad | Rajiv Gandhi International Stadium |
| Rajasthan Royals | Sawai Mansingh Stadium |
| Punjab Kings | Punjab Cricket Association Stadium |
| Lucknow Super Giants | BRSABV Ekana Cricket Stadium |
| Gujarat Titans | Narendra Modi Stadium |

## Dataset

Historical data sourced from Kaggle (2008–2025):
- `Datasets/matches.csv` — 1,170 match results
- `Datasets/deliveries.csv` — ~1.3M ball-by-ball records
- `Datasets/player_match_stats.csv` — per-player per-match statistics
- `Datasets/features_matrix.csv` — pre-computed feature matrix

## Project Structure

```
ipl-predictions-2026/
├── training.ipynb          # Model training pipeline
├── prediction.ipynb        # Match prediction interface
├── video_card.py           # Prediction card image generator
├── video_card_demo.py      # Demo card (no model needed)
├── project_paths.py        # Path configuration
├── requirements.txt        # Dependencies
├── Datasets/               # Historical IPL data (CSV)
├── Models/                 # Trained model artifacts (PKL)
└── output/                 # Generated prediction card images
```
