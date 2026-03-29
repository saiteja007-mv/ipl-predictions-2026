"""
predictor.py — Core prediction logic for IPL 2026 Predictor.

Provides clean, cached functions for loading models/data and computing predictions,
team form, H2H records, and venue statistics. Imported by app.py.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)
SEASON_TRACKER_PATH = OUTPUT_DIR / "season_tracker.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IPL_TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

IPL_VENUES = [
    "Wankhede Stadium",
    "MA Chidambaram Stadium",
    "M Chinnaswamy Stadium",
    "Eden Gardens",
    "Arun Jaitley Stadium",
    "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium",
    "Punjab Cricket Association Stadium",
    "BRSABV Ekana Cricket Stadium",
    "Narendra Modi Stadium",
]

HOME_CITY: dict[str, str] = {
    "Mumbai Indians": "Mumbai",
    "Chennai Super Kings": "Chennai",
    "Royal Challengers Bengaluru": "Bangalore",
    "Kolkata Knight Riders": "Kolkata",
    "Delhi Capitals": "Delhi",
    "Sunrisers Hyderabad": "Hyderabad",
    "Rajasthan Royals": "Jaipur",
    "Punjab Kings": "Mohali",
    "Lucknow Super Giants": "Lucknow",
    "Gujarat Titans": "Ahmedabad",
}

VENUE_CITY: dict[str, str] = {
    "Wankhede Stadium": "Mumbai",
    "MA Chidambaram Stadium": "Chennai",
    "M Chinnaswamy Stadium": "Bangalore",
    "Eden Gardens": "Kolkata",
    "Arun Jaitley Stadium": "Delhi",
    "Rajiv Gandhi International Stadium": "Hyderabad",
    "Sawai Mansingh Stadium": "Jaipur",
    "Punjab Cricket Association Stadium": "Mohali",
    "BRSABV Ekana Cricket Stadium": "Lucknow",
    "Narendra Modi Stadium": "Ahmedabad",
}

TEAM_COLORS: dict[str, str] = {
    "Mumbai Indians": "#004BA0",
    "Chennai Super Kings": "#FFFF3C",
    "Royal Challengers Bengaluru": "#D1001C",
    "Kolkata Knight Riders": "#3B0067",
    "Delhi Capitals": "#00008B",
    "Sunrisers Hyderabad": "#F26522",
    "Rajasthan Royals": "#EA1A85",
    "Punjab Kings": "#DCDDDF",
    "Lucknow Super Giants": "#A0E6FF",
    "Gujarat Titans": "#1C1C1C",
}

TEAM_ABBR: dict[str, str] = {
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC",
    "Sunrisers Hyderabad": "SRH",
    "Rajasthan Royals": "RR",
    "Punjab Kings": "PBKS",
    "Lucknow Super Giants": "LSG",
    "Gujarat Titans": "GT",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model() -> tuple:
    """Load the trained prediction model and feature metadata.

    Tries ensemble model first, falls back to XGBoost. Also loads
    feature_columns and team_name_map.

    Returns:
        (model, feature_columns, team_name_map) tuple, or (None, None, None)
        if no model is found.
    """
    model = None
    model_name = None

    for fname in ["ensemble_ipl_model.pkl", "xgb_ipl_model.pkl"]:
        path = MODEL_DIR / fname
        if path.exists():
            try:
                model = joblib.load(path)
                model_name = fname
                break
            except Exception:
                continue

    if model is None:
        return None, None, None

    feature_columns_path = MODEL_DIR / "feature_columns.pkl"
    team_name_map_path = MODEL_DIR / "team_name_map.pkl"

    feature_columns = joblib.load(feature_columns_path) if feature_columns_path.exists() else None
    team_name_map = joblib.load(team_name_map_path) if team_name_map_path.exists() else {}

    return model, feature_columns, team_name_map


def load_elo_ratings() -> dict:
    """Load pre-computed Elo ratings from disk.

    Returns:
        Dict mapping team name → Elo rating float. Empty dict if not found.
    """
    path = MODEL_DIR / "elo_ratings.pkl"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_enhanced_player_stats() -> pd.DataFrame:
    """Load player stats, preferring the global T20 enhanced version.

    Tries ``Datasets/player_stats_enhanced.csv`` first (produced by
    ``t20_data_pipeline.py``), falls back to ``Datasets/player_stats.csv``.
    The enhanced file contains global T20 career stats for all players,
    including those with zero IPL matches (new auction buys, uncapped players).

    Returns:
        Player stats DataFrame.  Empty DataFrame if neither file exists.
    """
    for fname in ("player_stats_enhanced.csv", "player_stats.csv"):
        path = DATASET_DIR / fname
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)
                if not df.empty:
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load core datasets: matches, player_match_stats, player_stats.

    Returns:
        (matches_df, player_match_stats_df, player_stats_df). Any missing
        dataset is returned as an empty DataFrame.  player_stats prefers
        the global T20 enhanced version if available.
    """
    def _read(fname: str) -> pd.DataFrame:
        path = DATASET_DIR / fname
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return pd.DataFrame()

    matches = _read("matches.csv")
    player_match_stats = _read("player_match_stats.csv")
    player_stats = load_enhanced_player_stats()

    # Normalize matches column names
    if not matches.empty:
        matches.columns = [c.strip().lower().replace(" ", "_") for c in matches.columns]
        if "match_date" in matches.columns and "date" not in matches.columns:
            matches = matches.rename(columns={"match_date": "date"})
        if "matchid" in matches.columns and "match_id" not in matches.columns:
            matches = matches.rename(columns={"matchid": "match_id"})
        if "date" in matches.columns:
            matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

        # Build id→name map from teams_data.csv and team_aliases.csv
        id_to_name = _build_id_to_name_map()
        team_name_norm = _build_team_name_map()

        def _resolve_team(x) -> str:
            """Resolve a team value that may be a numeric ID or a string name."""
            if pd.isna(x):
                return x
            s = str(x).strip()
            # Try numeric ID lookup first
            try:
                numeric_id = int(float(s))
                if numeric_id in id_to_name:
                    resolved = id_to_name[numeric_id]
                    return team_name_norm.get(resolved, resolved)
            except (ValueError, TypeError):
                pass
            # String name normalization
            return team_name_norm.get(s, s)

        for col in ["team1", "team2", "toss_winner", "match_winner"]:
            if col in matches.columns:
                matches[col] = matches[col].map(_resolve_team)

    return matches, player_match_stats, player_stats


def _build_id_to_name_map() -> dict[int, str]:
    """Build a mapping from numeric team ID to canonical team name.

    Reads teams_data.csv. Returns empty dict if file not found.

    Returns:
        Dict mapping int team_id → team_name string.
    """
    teams_path = DATASET_DIR / "teams_data.csv"
    if not teams_path.exists():
        return {}
    try:
        df = pd.read_csv(teams_path)
        return {int(row["team_id"]): str(row["team_name"]) for _, row in df.iterrows()}
    except Exception:
        return {}


def _build_team_name_map() -> dict[str, str]:
    """Build a comprehensive team name normalization map.

    Returns:
        Dict mapping raw team name variants → canonical name.
    """
    # Try loading saved map first
    saved_path = MODEL_DIR / "team_name_map.pkl"
    if saved_path.exists():
        try:
            return joblib.load(saved_path)
        except Exception:
            pass

    # Fallback hard-coded map
    return {
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
        "RCB": "Royal Challengers Bengaluru",
        "Mumbai Indians": "Mumbai Indians",
        "MI": "Mumbai Indians",
        "Chennai Super Kings": "Chennai Super Kings",
        "CSK": "Chennai Super Kings",
        "Kolkata Knight Riders": "Kolkata Knight Riders",
        "KKR": "Kolkata Knight Riders",
        "Delhi Capitals": "Delhi Capitals",
        "Delhi Daredevils": "Delhi Capitals",
        "DC": "Delhi Capitals",
        "Sunrisers Hyderabad": "Sunrisers Hyderabad",
        "SRH": "Sunrisers Hyderabad",
        "Rajasthan Royals": "Rajasthan Royals",
        "RR": "Rajasthan Royals",
        "Punjab Kings": "Punjab Kings",
        "Kings XI Punjab": "Punjab Kings",
        "PBKS": "Punjab Kings",
        "Lucknow Super Giants": "Lucknow Super Giants",
        "LSG": "Lucknow Super Giants",
        "Gujarat Titans": "Gujarat Titans",
        "GT": "Gujarat Titans",
        "Rising Pune Supergiant": "Rising Pune Supergiant",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
        "Pune Warriors": "Pune Warriors",
        "Deccan Chargers": "Deccan Chargers",
        "Kochi Tuskers Kerala": "Kochi Tuskers Kerala",
    }


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def get_team_form(team: str, matches: pd.DataFrame, n: int = 5) -> tuple[float, str]:
    """Compute a team's recent form over their last n matches.

    Args:
        team: Canonical team name.
        matches: Full matches DataFrame.
        n: Number of recent matches to consider.

    Returns:
        (win_rate, form_str) e.g. (0.8, "WWLWW")
    """
    if matches.empty:
        return 0.5, "?"

    team_matches = matches[
        (matches.get("team1", pd.Series(dtype=str)) == team) |
        (matches.get("team2", pd.Series(dtype=str)) == team)
    ].copy()

    if "date" in team_matches.columns:
        team_matches = team_matches.sort_values("date")

    team_matches = team_matches.tail(n)
    if team_matches.empty:
        return 0.5, "?"

    winner_col = "match_winner" if "match_winner" in team_matches.columns else "winner"
    results = []
    for _, row in team_matches.iterrows():
        winner = row.get(winner_col, None)
        if pd.isna(winner):
            results.append("?")
        elif str(winner).strip() == team:
            results.append("W")
        else:
            results.append("L")

    wins = results.count("W")
    total = len([r for r in results if r != "?"])
    win_rate = wins / total if total > 0 else 0.5
    form_str = "".join(results) if results else "?"
    return win_rate, form_str


def get_h2h(team1: str, team2: str, matches: pd.DataFrame) -> tuple[int, int, int]:
    """Compute head-to-head record between two teams.

    Args:
        team1: First team name.
        team2: Second team name.
        matches: Full matches DataFrame.

    Returns:
        (team1_wins, team2_wins, total_matches)
    """
    if matches.empty:
        return 0, 0, 0

    winner_col = "match_winner" if "match_winner" in matches.columns else "winner"
    h2h = matches[
        ((matches.get("team1") == team1) & (matches.get("team2") == team2)) |
        ((matches.get("team1") == team2) & (matches.get("team2") == team1))
    ]

    t1_wins = (h2h[winner_col] == team1).sum()
    t2_wins = (h2h[winner_col] == team2).sum()
    total = len(h2h)
    return int(t1_wins), int(t2_wins), int(total)


def get_venue_stats(venue: str, team1: str, team2: str, matches: pd.DataFrame) -> dict:
    """Compute venue-specific statistics for a match.

    Args:
        venue: Venue name.
        team1: First team name.
        team2: Second team name.
        matches: Full matches DataFrame.

    Returns:
        Dict with keys: total, bat_first_pct, team1_pct, team2_pct
    """
    if matches.empty or "venue" not in matches.columns:
        return {"total": 0, "bat_first_pct": 50.0, "team1_pct": 50.0, "team2_pct": 50.0}

    venue_matches = matches[matches["venue"].str.contains(venue.split(",")[0], na=False, case=False)]
    total = len(venue_matches)

    if total == 0:
        return {"total": 0, "bat_first_pct": 50.0, "team1_pct": 50.0, "team2_pct": 50.0}

    winner_col = "match_winner" if "match_winner" in matches.columns else "winner"

    # Bat-first win %
    bat_first = venue_matches[venue_matches.get("toss_decision", pd.Series()) == "bat"]
    if len(bat_first) > 0:
        toss_col = "toss_winner"
        bat_first_wins = (bat_first[winner_col] == bat_first[toss_col]).sum()
        bat_first_pct = round(bat_first_wins / len(bat_first) * 100, 1)
    else:
        bat_first_pct = 50.0

    # Team win % at venue
    t1_venue = venue_matches[(venue_matches.get("team1") == team1) | (venue_matches.get("team2") == team1)]
    t1_wins_v = (t1_venue[winner_col] == team1).sum()
    team1_pct = round(t1_wins_v / len(t1_venue) * 100, 1) if len(t1_venue) > 0 else 50.0

    t2_venue = venue_matches[(venue_matches.get("team1") == team2) | (venue_matches.get("team2") == team2)]
    t2_wins_v = (t2_venue[winner_col] == team2).sum()
    team2_pct = round(t2_wins_v / len(t2_venue) * 100, 1) if len(t2_venue) > 0 else 50.0

    return {
        "total": total,
        "bat_first_pct": bat_first_pct,
        "team1_pct": team1_pct,
        "team2_pct": team2_pct,
    }


def get_player_strength(
    team: str,
    player_match_stats: pd.DataFrame,
    enhanced_player_stats: Optional[pd.DataFrame] = None,
    team_xi: Optional[list[str]] = None,
) -> dict:
    """Compute aggregate player strength metrics for a team.

    For players with zero IPL matches (new auction buys, uncapped players),
    falls back to global T20 career stats from ``player_stats_enhanced.csv``
    instead of using league-average defaults.

    Args:
        team: Team name.
        player_match_stats: Per-player per-match stats DataFrame.
        enhanced_player_stats: Optional enhanced player stats DataFrame loaded
            from ``player_stats_enhanced.csv`` via ``load_enhanced_player_stats()``.
            When provided, used to fill in stats for players with no IPL history.
        team_xi: Optional list of player names in the playing XI.  When
            provided, strength is computed only over these players.

    Returns:
        Dict with keys: sr (batting SR), avg (batting avg), econ (bowling economy),
        exp (experience = unique players), bpct (boundary %)
    """
    defaults = {"sr": 130.0, "avg": 25.0, "econ": 8.0, "exp": 11, "bpct": 25.0}

    # --- IPL match stats path ---
    ipl_result = {"sr": None, "avg": None, "econ": None, "exp": None, "bpct": None}

    if not player_match_stats.empty and "team" in player_match_stats.columns:
        team_data = player_match_stats[player_match_stats["team"] == team]
        if team_xi:
            pc = next((c for c in team_data.columns if "player" in c.lower()), None)
            if pc:
                team_data = team_data[team_data[pc].isin(team_xi)]
        if not team_data.empty:
            ipl_result["sr"] = team_data["strike_rate"].mean() if "strike_rate" in team_data.columns else None
            ipl_result["avg"] = team_data["runs_scored"].mean() if "runs_scored" in team_data.columns else None
            ipl_result["econ"] = team_data["economy"].mean() if "economy" in team_data.columns else None
            ipl_result["exp"] = team_data["player"].nunique() if "player" in team_data.columns else None
            ipl_result["bpct"] = team_data["boundary_pct"].mean() if "boundary_pct" in team_data.columns else None

    # --- Enhanced / global T20 stats path for new players ---
    global_sr, global_avg, global_econ = None, None, None

    if enhanced_player_stats is not None and not enhanced_player_stats.empty:
        ep = enhanced_player_stats

        # If we have a specific XI, filter to those players only
        name_col = next(
            (c for c in ep.columns if c.lower() in ("player", "name", "player_name")),
            None,
        )
        if team_xi and name_col:
            ep_subset = ep[ep[name_col].isin(team_xi)]
        else:
            ep_subset = ep

        # Use global T20 batting SR for players that are new to IPL
        if "is_new_to_ipl" in ep_subset.columns and "t20_batting_sr" in ep_subset.columns:
            new_players = ep_subset[ep_subset["is_new_to_ipl"] == 1]
            if not new_players.empty:
                global_sr = new_players["t20_batting_sr"].mean()
                if "t20_batting_avg" in new_players.columns:
                    global_avg = new_players["t20_batting_avg"].mean()
                if "t20_bowling_econ" in new_players.columns:
                    global_econ = new_players["t20_bowling_econ"].mean()

    # --- Blend: use IPL stats where available, fill gaps with global T20 ---
    result = {}
    result["sr"] = ipl_result["sr"] if ipl_result["sr"] is not None and not pd.isna(ipl_result["sr"]) \
        else (global_sr if global_sr is not None else defaults["sr"])
    result["avg"] = ipl_result["avg"] if ipl_result["avg"] is not None and not pd.isna(ipl_result["avg"]) \
        else (global_avg if global_avg is not None else defaults["avg"])
    result["econ"] = ipl_result["econ"] if ipl_result["econ"] is not None and not pd.isna(ipl_result["econ"]) \
        else (global_econ if global_econ is not None else defaults["econ"])
    result["exp"] = ipl_result["exp"] if ipl_result["exp"] is not None else defaults["exp"]
    result["bpct"] = ipl_result["bpct"] if ipl_result["bpct"] is not None and not pd.isna(ipl_result["bpct"]) \
        else defaults["bpct"]

    # Final NaN guard
    for k, v in defaults.items():
        if pd.isna(result.get(k)):
            result[k] = v

    return result


def _elo_win_prob(team_elo: float, opponent_elo: float) -> float:
    """Compute expected win probability using Elo formula.

    Args:
        team_elo: Team's Elo rating.
        opponent_elo: Opponent's Elo rating.

    Returns:
        Expected win probability (0–1).
    """
    return 1.0 / (1.0 + 10 ** ((opponent_elo - team_elo) / 400))


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------


def build_features(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    matches: pd.DataFrame,
    player_match_stats: pd.DataFrame,
    feature_columns: list[str],
    elo_ratings: Optional[dict] = None,
    season: int = 2026,
    team1_xi: Optional[list[str]] = None,
    team2_xi: Optional[list[str]] = None,
    enhanced_player_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a feature vector for a match, aligned to model's expected columns.

    Args:
        team1: First team name.
        team2: Second team name.
        venue: Match venue.
        toss_winner: Team that won the toss.
        toss_decision: "bat" or "field".
        matches: Historical matches DataFrame.
        player_match_stats: Player per-match stats DataFrame.
        feature_columns: Ordered list of feature names from training.
        elo_ratings: Optional dict of team → Elo rating.
        season: Current season year.
        team1_xi: Optional list of playing XI names for team1.
        team2_xi: Optional list of playing XI names for team2.
        enhanced_player_stats: Optional enhanced player stats DataFrame from
            ``load_enhanced_player_stats()``.  Used to substitute global T20
            stats for players with no IPL match history.

    Returns:
        DataFrame with one row aligned to feature_columns.
    """
    if elo_ratings is None:
        elo_ratings = {}

    city = VENUE_CITY.get(venue, "")

    # Form & momentum
    t1_wr, _ = get_team_form(team1, matches, n=5)
    t2_wr, _ = get_team_form(team2, matches, n=5)

    # Season form (last 3 in current season)
    def season_form(team: str) -> float:
        if matches.empty or "season" not in matches.columns:
            return 0.5
        s_matches = matches[matches["season"] == season]
        s_team = s_matches[(s_matches.get("team1") == team) | (s_matches.get("team2") == team)]
        if s_team.empty:
            return 0.5
        s_team = s_team.tail(3)
        wc = "match_winner" if "match_winner" in s_team.columns else "winner"
        wins = (s_team[wc] == team).sum()
        return wins / len(s_team)

    t1_sf = season_form(team1)
    t2_sf = season_form(team2)

    # H2H
    t1_h2h, t2_h2h, h2h_total = get_h2h(team1, team2, matches)

    # Venue stats
    vstats = get_venue_stats(venue, team1, team2, matches)

    # Toss
    toss_is_t1 = 1 if toss_winner == team1 else 0
    toss_bat = 1 if toss_decision.lower() == "bat" else 0

    # Toss-venue interaction
    if not matches.empty and "venue" in matches.columns:
        venue_m = matches[matches["venue"].str.contains(venue.split(",")[0], na=False, case=False)]
        total_v = len(venue_m)
        if total_v > 0:
            wc = "match_winner" if "match_winner" in matches.columns else "winner"
            tw_col = "toss_winner"
            toss_v_wins = (venue_m[wc] == venue_m[tw_col]).sum() if tw_col in venue_m.columns else total_v * 0.5
            toss_venue_pct = toss_v_wins / total_v
        else:
            toss_venue_pct = 0.5
    else:
        toss_venue_pct = 0.5

    # Home advantage
    t1_home = 1 if HOME_CITY.get(team1, "") == city else 0
    t2_home = 1 if HOME_CITY.get(team2, "") == city else 0

    # Elo
    default_elo = 1500.0
    t1_elo = elo_ratings.get(team1, default_elo)
    t2_elo = elo_ratings.get(team2, default_elo)
    t1_elo_prob = _elo_win_prob(t1_elo, t2_elo)

    # Player strength — uses enhanced global T20 stats for new IPL players
    t1_ps = get_player_strength(team1, player_match_stats, enhanced_player_stats, team1_xi)
    t2_ps = get_player_strength(team2, player_match_stats, enhanced_player_stats, team2_xi)

    # Momentum (simplified: win_rate_last5 as proxy)
    t1_momentum = t1_wr
    t2_momentum = t2_wr

    # Build feature dict matching training columns
    feat: dict[str, float] = {
        "team1_win_rate_last5": t1_wr,
        "team2_win_rate_last5": t2_wr,
        "team1_season_form": t1_sf,
        "team2_season_form": t2_sf,
        "team1_momentum": t1_momentum,
        "team2_momentum": t2_momentum,
        "h2h_team1_wins": float(t1_h2h),
        "h2h_team2_wins": float(t2_h2h),
        "h2h_total": float(h2h_total),
        "venue_bat_first_win_pct": vstats["bat_first_pct"] / 100.0,
        "venue_team1_win_pct": vstats["team1_pct"] / 100.0,
        "venue_team2_win_pct": vstats["team2_pct"] / 100.0,
        "toss_winner_is_team1": float(toss_is_t1),
        "toss_decision_bat": float(toss_bat),
        "toss_venue_win_pct": toss_venue_pct,
        "team1_home": float(t1_home),
        "team2_home": float(t2_home),
        "team1_elo": t1_elo,
        "team2_elo": t2_elo,
        "elo_win_prob_team1": t1_elo_prob,
        "team1_avg_batting_sr": t1_ps["sr"],
        "team1_avg_batting_avg": t1_ps["avg"],
        "team1_avg_bowling_econ": t1_ps["econ"],
        "team1_avg_bowling_sr": t1_ps["sr"],
        "team1_total_experience": float(t1_ps["exp"]),
        "team1_new_player_count": 0.0,
        "team1_boundary_pct": t1_ps["bpct"],
        "team1_recent_batting_sr": t1_ps["sr"],
        "team1_recent_bowling_econ": t1_ps["econ"],
        "team2_avg_batting_sr": t2_ps["sr"],
        "team2_avg_batting_avg": t2_ps["avg"],
        "team2_avg_bowling_econ": t2_ps["econ"],
        "team2_avg_bowling_sr": t2_ps["sr"],
        "team2_total_experience": float(t2_ps["exp"]),
        "team2_new_player_count": 0.0,
        "team2_boundary_pct": t2_ps["bpct"],
        "team2_recent_batting_sr": t2_ps["sr"],
        "team2_recent_bowling_econ": t2_ps["econ"],
        "season": float(season),
        "is_playoff": 0.0,
    }

    # Align to feature_columns (fill any missing with 0)
    if feature_columns:
        row = {col: feat.get(col, 0.0) for col in feature_columns}
        return pd.DataFrame([row], columns=feature_columns)
    else:
        return pd.DataFrame([feat])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_match(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    model,
    feature_columns: list[str],
    matches: pd.DataFrame,
    player_match_stats: pd.DataFrame,
    elo_ratings: Optional[dict] = None,
    season: int = 2026,
    team1_xi: Optional[list[str]] = None,
    team2_xi: Optional[list[str]] = None,
) -> dict:
    """Run a full match prediction.

    Args:
        team1: First team name.
        team2: Second team name.
        venue: Match venue.
        toss_winner: Team that won the toss.
        toss_decision: "bat" or "field".
        model: Trained sklearn-compatible model.
        feature_columns: Feature column names.
        matches: Historical matches DataFrame.
        player_match_stats: Per-player stats DataFrame.
        elo_ratings: Optional Elo ratings dict.
        season: Season year.
        team1_xi: Optional list of playing XI names for team1.  When provided,
            player strength is computed over these specific players and global
            T20 stats are used for any players with no IPL history.
        team2_xi: Optional list of playing XI names for team2.

    Returns:
        Dict with keys: winner, prob_team1, prob_team2, confidence,
        form_team1, form_team2, h2h, venue_stats.
    """
    # Load enhanced player stats once for both teams
    enhanced_player_stats = load_enhanced_player_stats() if (team1_xi or team2_xi) else None

    X = build_features(
        team1, team2, venue, toss_winner, toss_decision,
        matches, player_match_stats, feature_columns, elo_ratings, season,
        team1_xi=team1_xi, team2_xi=team2_xi,
        enhanced_player_stats=enhanced_player_stats,
    )

    proba = model.predict_proba(X)[0]
    # model returns [prob_class0, prob_class1]; class1 = team1 wins
    if len(proba) == 2:
        prob_team1 = float(proba[1])
        prob_team2 = float(proba[0])
    else:
        prob_team1 = float(proba[0])
        prob_team2 = 1.0 - prob_team1

    winner = team1 if prob_team1 >= prob_team2 else team2
    confidence = max(prob_team1, prob_team2)

    _, form_team1 = get_team_form(team1, matches, n=5)
    _, form_team2 = get_team_form(team2, matches, n=5)
    h2h = get_h2h(team1, team2, matches)
    venue_stats = get_venue_stats(venue, team1, team2, matches)

    return {
        "winner": winner,
        "prob_team1": round(prob_team1, 4),
        "prob_team2": round(prob_team2, 4),
        "confidence": round(confidence, 4),
        "form_team1": form_team1,
        "form_team2": form_team2,
        "h2h": h2h,
        "venue_stats": venue_stats,
    }


# ---------------------------------------------------------------------------
# Season tracker
# ---------------------------------------------------------------------------


def load_season_tracker() -> dict:
    """Load season tracker from disk.

    Returns:
        Dict with key 'predictions' (list of prediction records).
    """
    if not SEASON_TRACKER_PATH.exists():
        return {"predictions": []}
    try:
        with open(SEASON_TRACKER_PATH) as f:
            return json.load(f)
    except Exception:
        return {"predictions": []}


def save_season_tracker(tracker: dict) -> None:
    """Persist season tracker to disk.

    Args:
        tracker: Dict with 'predictions' list.
    """
    with open(SEASON_TRACKER_PATH, "w") as f:
        json.dump(tracker, f, indent=2, default=str)


def add_prediction_to_tracker(
    tracker: dict,
    match_id: str,
    date: str,
    team1: str,
    team2: str,
    venue: str,
    predicted_winner: str,
    confidence: float,
) -> dict:
    """Append a new prediction record to the tracker.

    Args:
        tracker: Current tracker dict.
        match_id: Unique match identifier string.
        date: Match date string.
        team1: First team.
        team2: Second team.
        venue: Match venue.
        predicted_winner: Model's predicted winner.
        confidence: Prediction confidence (0–1).

    Returns:
        Updated tracker dict.
    """
    record = {
        "match_id": match_id,
        "date": date,
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "predicted_winner": predicted_winner,
        "confidence": round(confidence, 4),
        "actual_winner": None,
        "correct": None,
        "win_margin": None,
        "player_of_match": None,
    }
    tracker["predictions"].append(record)
    return tracker


def log_actual_result(
    tracker: dict,
    match_id: str,
    actual_winner: str,
    win_margin: str = "",
    player_of_match: str = "",
) -> tuple[dict, bool]:
    """Update a prediction record with the actual match result.

    Args:
        tracker: Current tracker dict.
        match_id: Match identifier to update.
        actual_winner: Team that actually won.
        win_margin: Optional margin description (e.g. "5 wickets").
        player_of_match: Optional player of the match name.

    Returns:
        (updated_tracker, found) where found is True if match_id was found.
    """
    found = False
    for record in tracker["predictions"]:
        if record["match_id"] == match_id:
            record["actual_winner"] = actual_winner
            record["correct"] = record["predicted_winner"] == actual_winner
            record["win_margin"] = win_margin
            record["player_of_match"] = player_of_match
            found = True
            break
    return tracker, found


def compute_accuracy(tracker: dict) -> tuple[int, int, float]:
    """Compute prediction accuracy from tracker.

    Args:
        tracker: Tracker dict with predictions list.

    Returns:
        (correct_count, total_logged, accuracy_pct)
    """
    logged = [p for p in tracker["predictions"] if p.get("correct") is not None]
    correct = sum(1 for p in logged if p["correct"])
    total = len(logged)
    accuracy = round(correct / total * 100, 1) if total > 0 else 0.0
    return correct, total, accuracy


# ---------------------------------------------------------------------------
# Stats explorer helpers
# ---------------------------------------------------------------------------


def get_team_all_time_stats(team: str, matches: pd.DataFrame) -> dict:
    """Compute all-time win statistics for a team.

    Args:
        team: Team name.
        matches: Full matches DataFrame.

    Returns:
        Dict with keys: total_matches, wins, win_rate
    """
    if matches.empty:
        return {"total_matches": 0, "wins": 0, "win_rate": 0.0}

    team_m = matches[(matches.get("team1") == team) | (matches.get("team2") == team)]
    wc = "match_winner" if "match_winner" in matches.columns else "winner"
    wins = (team_m[wc] == team).sum()
    total = len(team_m)
    return {
        "total_matches": total,
        "wins": int(wins),
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
    }


def get_team_recent_form_series(team: str, matches: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get last n match results for a team as a DataFrame.

    Args:
        team: Team name.
        matches: Full matches DataFrame.
        n: Number of matches to return.

    Returns:
        DataFrame with columns: date, opponent, result (W/L), venue
    """
    if matches.empty:
        return pd.DataFrame()

    team_m = matches[
        (matches.get("team1") == team) | (matches.get("team2") == team)
    ].copy()

    if "date" in team_m.columns:
        team_m = team_m.sort_values("date")

    team_m = team_m.tail(n)
    wc = "match_winner" if "match_winner" in matches.columns else "winner"

    rows = []
    for _, row in team_m.iterrows():
        opponent = row["team2"] if row.get("team1") == team else row.get("team1", "?")
        result = "W" if str(row.get(wc, "")).strip() == team else "L"
        rows.append({
            "date": row.get("date", ""),
            "opponent": opponent,
            "result": result,
            "venue": row.get("venue", ""),
        })

    return pd.DataFrame(rows)


def get_h2h_matrix(matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Build a head-to-head win count matrix for a list of teams.

    Args:
        matches: Full matches DataFrame.
        teams: List of team names.

    Returns:
        DataFrame (index=teams, columns=teams) with win counts.
    """
    matrix = pd.DataFrame(0, index=teams, columns=teams, dtype=int)
    if matches.empty:
        return matrix

    wc = "match_winner" if "match_winner" in matches.columns else "winner"
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            h2h_m = matches[
                ((matches.get("team1") == t1) & (matches.get("team2") == t2)) |
                ((matches.get("team1") == t2) & (matches.get("team2") == t1))
            ]
            wins = (h2h_m[wc] == t1).sum()
            matrix.loc[t1, t2] = int(wins)
    return matrix


def search_player_stats(player_query: str, player_stats: pd.DataFrame) -> pd.DataFrame:
    """Search player stats by name (case-insensitive partial match).

    Args:
        player_query: Search string.
        player_stats: Player stats DataFrame.

    Returns:
        Filtered DataFrame of matching players.
    """
    if player_stats.empty or not player_query.strip():
        return pd.DataFrame()

    name_col = next((c for c in player_stats.columns if "player" in c.lower() or "name" in c.lower()), None)
    if name_col is None:
        return pd.DataFrame()

    mask = player_stats[name_col].astype(str).str.contains(player_query.strip(), case=False, na=False)
    return player_stats[mask].head(20)
