# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda create -n ipl python=3.11 -y
conda activate ipl
pip install -r requirements.txt
```

**Windows note:** `numpy>=2.2` has a broken install on Windows; `requirements.txt` pins `numpy<2.2`.

**Tesseract binary** (for OCR features):
```bash
sudo apt-get install tesseract-ocr   # Linux
brew install tesseract               # macOS
# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Running the App

```bash
# Run the Streamlit web app
streamlit run app.py

# Generate a demo prediction card (no model needed)
python video_card_demo.py

# Build global T20 player stats (first time: ~200 MB download)
python t20_data_pipeline.py --download
# Subsequent runs without re-downloading:
python t20_data_pipeline.py --no-parse
```

## Project Root Path

If Jupyter's CWD differs from the project folder, set:
```powershell
$env:IPL_PROJECT_ROOT = 'D:/Projects/IPL 2026'
```
`project_paths.py` reads this env var; `predictor.py` and `xi_extractor.py` use `Path(__file__).parent` directly.

## Architecture

### Data Flow

```
Datasets/matches.csv + deliveries.csv
        ↓
training.ipynb  →  feature engineering  →  Models/*.pkl
        ↓
prediction.ipynb / app.py  →  predictor.py  →  ensemble prediction
```

### Key Files

| File | Role |
|------|------|
| `predictor.py` | Core prediction engine — loads models/data, computes features, exposes `predict_match()`. Imported by `app.py`. |
| `app.py` | Streamlit 4-page UI: Predict Match, Season Tracker, Log Match Result, Stats Explorer |
| `xi_extractor.py` | Playing XI extraction from images via Tesseract OCR → Ollama LLM → regex/fuzzy fallback |
| `t20_data_pipeline.py` | Downloads cricsheet YAML data and builds `player_stats_enhanced.csv` (global T20 career stats) |
| `video_card.py` | Generates PNG prediction cards (matplotlib) |
| `training.ipynb` | Full training pipeline: data download, EDA, feature engineering, model training, save artifacts |
| `prediction.ipynb` | Standalone notebook for batch prediction and post-match dataset updates |

### Model Ensemble

Three models trained on IPL 2008–2024, tested on 2025, saved to `Models/`:
- `lgbm_ipl_model.pkl` — LightGBM (primary)
- `catboost_ipl_model.pkl` — CatBoost (handles team-name categoricals)
- `xgb_ipl_model.pkl` — XGBoost (baseline)
- `ensemble_ipl_model.pkl` — Soft-voting ensemble (~72–75% accuracy)
- `feature_columns.pkl`, `team_name_map.pkl`, `elo_ratings.pkl` — supporting artifacts

### Feature Engineering (no leakage)

All features are computed using only data available **before** the match:
- **Elo ratings** — rolling team strength updated after every match
- **Team form** — last 5 matches overall + last 3 this season
- **Head-to-head** — all-time record between the two teams
- **Venue stats** — bat-first win %, per-team venue win %
- **Player strength** — batting SR, bowling economy aggregated to team level
- **New player handling** — zero-IPL-history players get league-average defaults

### Player Stats Priority

`predictor.py` and `xi_extractor.py` prefer `Datasets/player_stats_enhanced.csv` (global T20 stats from cricsheet pipeline) over `Datasets/player_stats.csv` when the file exists.

## Ollama Integration (Playing XI from Photos)

`xi_extractor.py` connects to Ollama at `http://127.0.0.1:11434` (default) or `OLLAMA_HOST` env var. The health check hits `/api/tags` — **do not add `/v1` to the base URL**.

```bash
ollama pull llama3.2          # text model (uses Tesseract OCR first)
ollama pull llava             # vision model (reads image directly)
ollama serve                  # starts API on port 11434
```

If Ollama is unreachable the extractor silently falls back to regex + rapidfuzz matching against the known player list.

## Updating the Dataset After Matches

Add completed match results in `prediction.ipynb` Section 4, then restart the kernel so `predictor.py` reloads the updated CSVs. For the model to learn from 2026 data, re-run `training.ipynb` after adding results.
