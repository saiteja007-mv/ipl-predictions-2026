"""
t20_data_pipeline.py — Global T20 Player Data Pipeline for IPL 2026 Predictor.

Downloads ball-by-ball T20 data from cricsheet.org (free, open data) for all
international T20Is and major domestic T20 leagues, then builds a unified
player stats table that supplements IPL-specific data for new/uncapped players.

Usage:
    python t20_data_pipeline.py                # build stats (no download)
    python t20_data_pipeline.py --download     # download latest cricsheet data first
    python t20_data_pipeline.py --download --parse  # download + parse + build

Output:
    Datasets/cricsheet/                     raw YAML files
    Datasets/player_stats_enhanced.csv      global T20 career stats
    Datasets/player_name_map.csv            name normalisation lookup
"""

from __future__ import annotations

import argparse
import io
import os
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests

    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import yaml

    YAML_OK = True
except ImportError:
    YAML_OK = False

try:
    from rapidfuzz import process as rfuzz_process
    from rapidfuzz import fuzz

    RAPIDFUZZ_OK = True
except ImportError:
    RAPIDFUZZ_OK = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "Datasets"
CRICSHEET_DIR = DATASET_DIR / "cricsheet"

CRICSHEET_URL = "https://cricsheet.org/downloads/t20s_male.zip"
ENHANCED_STATS_PATH = DATASET_DIR / "player_stats_enhanced.csv"
NAME_MAP_PATH = DATASET_DIR / "player_name_map.csv"
IPL_DELIVERIES_PATH = DATASET_DIR / "deliveries.csv"
IPL_PLAYER_STATS_PATH = DATASET_DIR / "player_stats.csv"

# Columns produced by parse_cricsheet_yaml / build_global_player_stats
DELIVERY_COLS = [
    "match_id",
    "date",
    "format",
    "competition",
    "team",
    "player",
    "runs_scored",
    "balls_faced",
    "wickets_taken",
    "runs_conceded",
    "balls_bowled",
    "is_wicket",
    "dismissed_by",
]

GLOBAL_STATS_COLS = [
    "player",
    "t20_matches",
    "t20_runs",
    "t20_batting_sr",
    "t20_batting_avg",
    "t20_wickets",
    "t20_bowling_econ",
    "t20_bowling_sr",
    "ipl_matches",
    "global_experience",
    "is_new_to_ipl",
]


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------


def normalize_player_name(name: str) -> str:
    """Normalize a player name for cross-dataset matching.

    Steps:
    1. Unicode NFC normalisation → strip accents
    2. Lower-case and strip whitespace
    3. Collapse multiple spaces
    4. Remove punctuation except hyphens (for double-barrelled names)

    Args:
        name: Raw player name string.

    Returns:
        Normalised lower-case ASCII-ish name.
    """
    if not isinstance(name, str):
        return ""

    # NFC then strip combining characters (accents)
    nfc = unicodedata.normalize("NFC", name)
    ascii_approx = "".join(
        c for c in unicodedata.normalize("NFD", nfc) if unicodedata.category(c) != "Mn"
    )

    cleaned = ascii_approx.lower().strip()
    # Remove everything except letters, spaces, hyphens
    cleaned = re.sub(r"[^a-z\s\-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_name_map_from_csv() -> dict[str, str]:
    """Load existing name map from Datasets/player_name_map.csv.

    Returns:
        Dict mapping raw name → canonical name.
    """
    if not NAME_MAP_PATH.exists():
        return {}
    try:
        df = pd.read_csv(NAME_MAP_PATH)
        if "raw_name" in df.columns and "canonical_name" in df.columns:
            return dict(zip(df["raw_name"].astype(str), df["canonical_name"].astype(str)))
    except Exception:
        pass
    return {}


def fuzzy_resolve_name(
    name: str,
    canonical_names: list[str],
    threshold: float = 85.0,
    name_map: Optional[dict[str, str]] = None,
) -> str:
    """Resolve a raw name to a canonical name via exact match then fuzzy match.

    Args:
        name: Raw player name to resolve.
        canonical_names: List of known canonical names.
        threshold: Minimum rapidfuzz ratio to accept (0–100).
        name_map: Optional pre-built {raw → canonical} dict for fast lookup.

    Returns:
        Canonical name if matched, else the original ``name`` unchanged.
    """
    if name_map and name in name_map:
        return name_map[name]

    norm = normalize_player_name(name)
    norm_canonical = [normalize_player_name(c) for c in canonical_names]

    # Exact normalised match
    if norm in norm_canonical:
        return canonical_names[norm_canonical.index(norm)]

    # Fuzzy match
    if RAPIDFUZZ_OK and canonical_names:
        result = rfuzz_process.extractOne(norm, norm_canonical, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            return canonical_names[result[2]]

    return name


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def fetch_cricsheet_data(output_dir: str = str(CRICSHEET_DIR)) -> dict[str, int]:
    """Download T20 ball-by-ball data from cricsheet.org.

    Downloads t20s_male.zip (all international T20s as YAML), extracts YAML
    files into ``output_dir``.  Skips download if files are already present
    unless you delete the directory first.

    Args:
        output_dir: Directory to extract YAML files into.

    Returns:
        Dict ``{format: file_count}`` where ``format`` is derived from filenames.

    Raises:
        ImportError: If ``requests`` is not installed.
        RuntimeError: If the download fails.
    """
    if not REQUESTS_OK:
        raise ImportError(
            "requests is required to download cricsheet data. "
            "Install with: pip install requests"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {CRICSHEET_URL} …")
    try:
        response = requests.get(CRICSHEET_URL, stream=True, timeout=120)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to download cricsheet data: {exc}") from exc

    total_bytes = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunks: list[bytes] = []
    for chunk in response.iter_content(chunk_size=1024 * 256):
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = downloaded / total_bytes * 100
                print(f"\r  {pct:.1f}%  ({downloaded // 1024 // 1024} MB)", end="", flush=True)

    print()
    zip_bytes = b"".join(chunks)
    print(f"Downloaded {len(zip_bytes) // 1024 // 1024} MB. Extracting…")

    counts: dict[str, int] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            if member.endswith((".yaml", ".json")):
                zf.extract(member, out)
                ext = member.split(".")[-1]
                counts[ext] = counts.get(ext, 0) + 1

    print(f"Extracted: {counts}")
    return counts


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_single_yaml(path: Path, match_id: str) -> list[dict]:
    """Parse a single cricsheet YAML match file into delivery rows.

    Args:
        path: Path to the YAML file.
        match_id: Identifier string (usually the filename stem).

    Returns:
        List of dicts with keys matching DELIVERY_COLS.
    """
    if not YAML_OK:
        return []

    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    meta = data.get("info", {})
    date_val = ""
    dates = meta.get("dates", [])
    if dates:
        date_val = str(dates[0])

    competition = meta.get("competition", meta.get("event", {}).get("name", ""))
    if isinstance(competition, dict):
        competition = competition.get("name", "")

    match_type = meta.get("match_type", "T20")

    rows: list[dict] = []
    innings_list = data.get("innings", [])

    for inning in innings_list:
        if not isinstance(inning, dict):
            continue

        # Support both cricsheet schema versions
        batting_team = inning.get("team", "")
        overs_data = inning.get("overs", [])

        # Per-over per-delivery accumulation
        batter_balls: dict[str, int] = {}
        batter_runs: dict[str, int] = {}
        bowler_balls: dict[str, int] = {}
        bowler_runs: dict[str, int] = {}
        bowler_wickets: dict[str, int] = {}
        batter_dismissed_by: dict[str, str] = {}

        for over_entry in overs_data:
            if not isinstance(over_entry, dict):
                continue
            deliveries = over_entry.get("deliveries", [])
            for delivery in deliveries:
                if not isinstance(delivery, dict):
                    continue

                batter = delivery.get("batter", delivery.get("batsman", ""))
                bowler = delivery.get("bowler", "")
                runs_info = delivery.get("runs", {})
                batter_run = int(runs_info.get("batter", runs_info.get("batsman", 0)))
                total_run = int(runs_info.get("total", 0))
                extras = delivery.get("extras", {})
                is_wides = "wides" in extras
                is_no_ball = "noballs" in extras

                legal_delivery = not is_wides

                wickets = delivery.get("wickets", [])
                is_wicket = 0
                dismissal_fielder = ""
                for w in wickets:
                    if isinstance(w, dict) and w.get("player_out", "") == batter:
                        is_wicket = 1
                        fielders = w.get("fielders", [])
                        if fielders and isinstance(fielders[0], dict):
                            dismissal_fielder = fielders[0].get("name", "")

                # Accumulate batter stats
                if batter:
                    batter_balls[batter] = batter_balls.get(batter, 0) + (1 if legal_delivery else 0)
                    batter_runs[batter] = batter_runs.get(batter, 0) + batter_run
                    if is_wicket:
                        batter_dismissed_by[batter] = bowler

                # Accumulate bowler stats (legal deliveries only)
                if bowler:
                    bowler_balls[bowler] = bowler_balls.get(bowler, 0) + (
                        1 if not is_wides and not is_no_ball else 0
                    )
                    bowler_runs[bowler] = bowler_runs.get(bowler, 0) + total_run
                    if is_wicket and wickets:
                        kind = wickets[0].get("kind", "") if isinstance(wickets[0], dict) else ""
                        if kind not in ("run out", "retired hurt", "obstructing the field"):
                            bowler_wickets[bowler] = bowler_wickets.get(bowler, 0) + 1

        # Emit one row per player who batted or bowled
        all_players = set(batter_balls) | set(bowler_balls)
        for player in all_players:
            rows.append(
                {
                    "match_id": match_id,
                    "date": date_val,
                    "format": match_type,
                    "competition": competition,
                    "team": batting_team,
                    "player": player,
                    "runs_scored": batter_runs.get(player, 0),
                    "balls_faced": batter_balls.get(player, 0),
                    "wickets_taken": bowler_wickets.get(player, 0),
                    "runs_conceded": bowler_runs.get(player, 0),
                    "balls_bowled": bowler_balls.get(player, 0),
                    "is_wicket": 1 if player in batter_dismissed_by else 0,
                    "dismissed_by": batter_dismissed_by.get(player, ""),
                }
            )

    return rows


def parse_cricsheet_yaml(cricsheet_dir: str = str(CRICSHEET_DIR)) -> pd.DataFrame:
    """Parse all cricsheet YAML files into a unified deliveries DataFrame.

    Args:
        cricsheet_dir: Directory containing extracted YAML/JSON match files.

    Returns:
        DataFrame with columns matching DELIVERY_COLS.
        Returns empty DataFrame if directory not found or PyYAML not installed.
    """
    if not YAML_OK:
        print(
            "WARNING: PyYAML not installed. Install with: pip install pyyaml\n"
            "Returning empty DataFrame."
        )
        return pd.DataFrame(columns=DELIVERY_COLS)

    cdir = Path(cricsheet_dir)
    if not cdir.exists():
        print(f"WARNING: cricsheet directory not found: {cdir}")
        return pd.DataFrame(columns=DELIVERY_COLS)

    yaml_files = list(cdir.rglob("*.yaml")) + list(cdir.rglob("*.yml"))
    if not yaml_files:
        print(f"No YAML files found in {cdir}")
        return pd.DataFrame(columns=DELIVERY_COLS)

    print(f"Parsing {len(yaml_files)} YAML files…")
    all_rows: list[dict] = []
    for i, fpath in enumerate(yaml_files):
        if i % 500 == 0:
            print(f"  {i}/{len(yaml_files)}", end="\r", flush=True)
        rows = _parse_single_yaml(fpath, fpath.stem)
        all_rows.extend(rows)

    print(f"\nParsed {len(all_rows)} delivery-level rows from {len(yaml_files)} files.")
    df = pd.DataFrame(all_rows, columns=DELIVERY_COLS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Stats building
# ---------------------------------------------------------------------------


def build_global_player_stats(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player global T20 career stats from all formats combined.

    Args:
        deliveries_df: DataFrame produced by ``parse_cricsheet_yaml`` with
            columns matching DELIVERY_COLS.

    Returns:
        DataFrame with columns matching GLOBAL_STATS_COLS.
        Also includes ``ipl_matches`` drawn from ``deliveries.csv`` if present.
    """
    if deliveries_df.empty:
        return pd.DataFrame(columns=GLOBAL_STATS_COLS)

    # --- Matches per player (for experience count) ---
    played = deliveries_df.groupby("player")["match_id"].nunique().rename("t20_matches")

    # --- Batting ---
    batting = (
        deliveries_df[deliveries_df["balls_faced"] > 0]
        .groupby("player")
        .agg(
            total_runs=("runs_scored", "sum"),
            total_balls=("balls_faced", "sum"),
            innings_count=("match_id", "nunique"),
            dismissals=("is_wicket", "sum"),
        )
    )
    batting["t20_runs"] = batting["total_runs"]
    batting["t20_batting_sr"] = np.where(
        batting["total_balls"] > 0,
        batting["total_runs"] / batting["total_balls"] * 100,
        0.0,
    )
    batting["t20_batting_avg"] = np.where(
        batting["dismissals"] > 0,
        batting["total_runs"] / batting["dismissals"],
        batting["total_runs"],  # not out average = total runs
    )

    # --- Bowling ---
    bowling = (
        deliveries_df[deliveries_df["balls_bowled"] > 0]
        .groupby("player")
        .agg(
            wkts=("wickets_taken", "sum"),
            conceded=("runs_conceded", "sum"),
            balls=("balls_bowled", "sum"),
        )
    )
    bowling["t20_wickets"] = bowling["wkts"]
    bowling["t20_bowling_econ"] = np.where(
        bowling["balls"] > 0, bowling["conceded"] / bowling["balls"] * 6, 0.0
    )
    bowling["t20_bowling_sr"] = np.where(
        bowling["wkts"] > 0, bowling["balls"] / bowling["wkts"], 0.0
    )

    # --- Merge all ---
    stats = pd.DataFrame(index=played.index)
    stats["t20_matches"] = played
    stats = stats.join(batting[["t20_runs", "t20_batting_sr", "t20_batting_avg"]], how="left")
    stats = stats.join(bowling[["t20_wickets", "t20_bowling_econ", "t20_bowling_sr"]], how="left")
    stats = stats.fillna(
        {
            "t20_runs": 0,
            "t20_batting_sr": 0.0,
            "t20_batting_avg": 0.0,
            "t20_wickets": 0,
            "t20_bowling_econ": 0.0,
            "t20_bowling_sr": 0.0,
        }
    )

    # --- IPL match count from existing deliveries.csv ---
    ipl_counts: pd.Series = pd.Series(dtype=int)
    if IPL_DELIVERIES_PATH.exists():
        try:
            ipl_del = pd.read_csv(IPL_DELIVERIES_PATH, usecols=lambda c: c in ("batter", "batsman", "match_id"), low_memory=False)
            batter_col = "batter" if "batter" in ipl_del.columns else "batsman"
            if batter_col in ipl_del.columns:
                ipl_counts = ipl_del.groupby(batter_col)["match_id"].nunique()
        except Exception:
            pass

    stats["ipl_matches"] = ipl_counts.reindex(stats.index).fillna(0).astype(int)
    stats["global_experience"] = stats["t20_matches"]
    stats["is_new_to_ipl"] = (stats["ipl_matches"] == 0).astype(int)
    stats = stats.reset_index().rename(columns={"player": "player"})

    # Round floats
    for col in ["t20_batting_sr", "t20_batting_avg", "t20_bowling_econ", "t20_bowling_sr"]:
        stats[col] = stats[col].round(2)

    return stats[GLOBAL_STATS_COLS]


# ---------------------------------------------------------------------------
# Merging with IPL stats
# ---------------------------------------------------------------------------


def merge_with_ipl_stats(
    global_stats: pd.DataFrame,
    ipl_player_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Merge global T20 stats with existing IPL player stats.

    Priority rules:
    - For IPL-experienced players (ipl_matches > 0): keep IPL stats as primary,
      append global T20 columns as ``_global`` suffix.
    - For new IPL players (ipl_matches == 0): use global T20 stats to fill
      IPL stat columns instead of league-average defaults.

    Args:
        global_stats: DataFrame from ``build_global_player_stats``.
        ipl_player_stats: DataFrame loaded from ``player_stats.csv``.

    Returns:
        Enhanced player stats DataFrame saved to
        ``Datasets/player_stats_enhanced.csv``.
    """
    if global_stats.empty:
        return ipl_player_stats.copy() if not ipl_player_stats.empty else pd.DataFrame()

    # Normalise names in both DataFrames
    global_stats = global_stats.copy()
    global_stats["player_norm"] = global_stats["player"].apply(normalize_player_name)

    if ipl_player_stats.empty:
        enhanced = global_stats.copy()
        enhanced.to_csv(ENHANCED_STATS_PATH, index=False)
        return enhanced

    ipl = ipl_player_stats.copy()

    # Detect name column
    name_col = next(
        (c for c in ipl.columns if c.lower() in ("player", "name", "player_name")),
        None,
    )
    if name_col is None:
        print("WARNING: Could not find player name column in ipl_player_stats.")
        enhanced = global_stats.copy()
        enhanced.to_csv(ENHANCED_STATS_PATH, index=False)
        return enhanced

    ipl["player_norm"] = ipl[name_col].astype(str).apply(normalize_player_name)

    # Merge on normalised name
    merged = ipl.merge(
        global_stats.add_suffix("_global").rename(columns={"player_norm_global": "player_norm"}),
        on="player_norm",
        how="outer",
    )

    # For new IPL players: fill batting/bowling with global T20 values
    new_mask = merged.get("ipl_matches_global", pd.Series(dtype=int)) > 0
    ipl_match_col = next((c for c in ipl.columns if "match" in c.lower()), None)

    if ipl_match_col:
        experienced_mask = merged[ipl_match_col].fillna(0) > 0
        new_to_ipl_mask = ~experienced_mask

        for src_col, dst_col in [
            ("t20_batting_sr_global", None),
            ("t20_batting_avg_global", None),
            ("t20_bowling_econ_global", None),
        ]:
            if src_col in merged.columns:
                # For new players, use global stats where IPL stats are missing
                for check_col in [c for c in ipl.columns if any(k in c.lower() for k in ("sr", "avg", "econ", "economy"))]:
                    if check_col in merged.columns:
                        merged.loc[new_to_ipl_mask & merged[check_col].isna(), check_col] = (
                            merged.loc[new_to_ipl_mask & merged[check_col].isna(), src_col]
                        )

    merged = merged.drop(columns=["player_norm"], errors="ignore")
    merged.to_csv(ENHANCED_STATS_PATH, index=False)
    print(f"Saved enhanced stats to {ENHANCED_STATS_PATH} ({len(merged)} players)")
    return merged


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def update_pipeline(
    download: bool = False,
    parse: bool = True,
) -> pd.DataFrame:
    """Run the full data pipeline to produce player_stats_enhanced.csv.

    Args:
        download: If True, download latest cricsheet data first.
        parse: If True, parse YAML files and rebuild global stats.
            If False, tries to load a previously parsed cache.

    Returns:
        Enhanced player stats DataFrame.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    if download:
        fetch_cricsheet_data()

    # Step 2: Parse YAML → deliveries DataFrame
    if parse:
        deliveries_df = parse_cricsheet_yaml()
    else:
        # Try to load cached parsed deliveries
        cached_path = DATASET_DIR / "cricsheet_deliveries_cache.csv"
        if cached_path.exists():
            print(f"Loading cached deliveries from {cached_path}")
            deliveries_df = pd.read_csv(cached_path, low_memory=False)
        else:
            print("No cache found — parsing YAML files…")
            deliveries_df = parse_cricsheet_yaml()

    # Cache deliveries for fast re-runs
    if not deliveries_df.empty:
        cache_path = DATASET_DIR / "cricsheet_deliveries_cache.csv"
        deliveries_df.to_csv(cache_path, index=False)
        print(f"Delivery cache saved to {cache_path}")

    # Step 3: Build global player stats
    print("Building global player stats…")
    global_stats = build_global_player_stats(deliveries_df)

    # Step 4: Load existing IPL player stats
    ipl_stats = pd.DataFrame()
    if IPL_PLAYER_STATS_PATH.exists():
        try:
            ipl_stats = pd.read_csv(IPL_PLAYER_STATS_PATH, low_memory=False)
            print(f"Loaded IPL player stats: {len(ipl_stats)} records")
        except Exception as exc:
            print(f"Could not load player_stats.csv: {exc}")

    # Step 5: Merge and save
    enhanced = merge_with_ipl_stats(global_stats, ipl_stats)

    # Save name map for debugging
    name_map_df = pd.DataFrame(
        {
            "raw_name": global_stats["player"].tolist(),
            "canonical_name": global_stats["player"].tolist(),
            "normalized": global_stats["player"].apply(normalize_player_name).tolist(),
        }
    )
    name_map_df.to_csv(NAME_MAP_PATH, index=False)
    print(f"Name map saved to {NAME_MAP_PATH}")

    print("Pipeline complete.")
    return enhanced


# ---------------------------------------------------------------------------
# Public helper used by predictor.py
# ---------------------------------------------------------------------------


def load_enhanced_stats() -> pd.DataFrame:
    """Load player_stats_enhanced.csv, falling back to player_stats.csv.

    Returns:
        Player stats DataFrame (enhanced if available, else IPL-only).
    """
    if ENHANCED_STATS_PATH.exists():
        try:
            df = pd.read_csv(ENHANCED_STATS_PATH, low_memory=False)
            if not df.empty:
                return df
        except Exception:
            pass

    if IPL_PLAYER_STATS_PATH.exists():
        try:
            return pd.read_csv(IPL_PLAYER_STATS_PATH, low_memory=False)
        except Exception:
            pass

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IPL 2026 — Global T20 Player Data Pipeline"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download latest cricsheet T20 data before building stats.",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        default=True,
        help="Parse YAML files and rebuild stats (default: True).",
    )
    parser.add_argument(
        "--no-parse",
        dest="parse",
        action="store_false",
        help="Skip YAML parsing; use cached deliveries CSV if available.",
    )
    args = parser.parse_args()
    update_pipeline(download=args.download, parse=args.parse)
