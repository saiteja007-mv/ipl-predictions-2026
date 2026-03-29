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

## Global T20 Player Data Pipeline

New and uncapped IPL players have no IPL history, so the model previously used
league-average defaults for their stats.  `t20_data_pipeline.py` fixes this by
pulling career stats from **all T20 formats** (international T20Is, BBL, PSL,
SA20, CPL, etc.) via [cricsheet.org](https://cricsheet.org/downloads/) open data.

### Run the pipeline

```bash
# First time: download ~200 MB of cricsheet YAML data and build stats
python t20_data_pipeline.py --download

# Subsequent runs: skip download, rebuild stats from cached data
python t20_data_pipeline.py --no-parse   # use cached deliveries CSV
python t20_data_pipeline.py              # re-parse YAML files
```

Output files:
- `Datasets/cricsheet/` — raw YAML match files from cricsheet.org
- `Datasets/player_stats_enhanced.csv` — global T20 career stats for all players
- `Datasets/player_name_map.csv` — name normalisation lookup table

The app and predictor automatically prefer `player_stats_enhanced.csv` over
`player_stats.csv` when it exists.

### Dependencies

```bash
pip install requests pyyaml rapidfuzz
```

PyYAML is required for YAML parsing.

---

## Playing XI from Photo (OCR)

On the **🏏 Predict Match** page you can now supply the playing XI by:

1. **Typing names manually** — one player per line
2. **Uploading a screenshot** of the XI announcement (IPL app, Twitter, TV graphic)

The photo upload uses Tesseract OCR to extract text, then optionally a local
[Ollama](https://ollama.com/) LLM (`llama3.2` or similar) to parse player names,
with a regex + fuzzy-match fallback when Ollama is not running.

### Setup OCR

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Python packages (already in requirements.txt)
pip install pytesseract Pillow rapidfuzz
```

### Optional: Ollama for better name extraction

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2   # or mistral, llama3, etc.
ollama serve           # start local API on port 11434
```

If Ollama is not running, the extractor falls back to regex + fuzzy matching
against the player database — no internet connection required.

---

## Project Structure

```
ipl-predictions-2026/
├── training.ipynb              # Model training pipeline
├── prediction.ipynb            # Match prediction interface
├── app.py                      # Streamlit web app (4 pages)
├── predictor.py                # Core prediction logic
├── t20_data_pipeline.py        # Global T20 player data pipeline
├── xi_extractor.py             # Playing XI extraction from images (OCR)
├── video_card.py               # Prediction card image generator
├── video_card_demo.py          # Demo card (no model needed)
├── project_paths.py            # Path configuration
├── requirements.txt            # Dependencies
├── Datasets/                   # Historical IPL data (CSV)
│   ├── cricsheet/              # Raw cricsheet YAML files (after --download)
│   ├── player_stats_enhanced.csv  # Global T20 stats (after pipeline run)
│   └── player_name_map.csv    # Name normalisation table
├── Models/                     # Trained model artifacts (PKL)
└── output/                     # Generated prediction card images
```
