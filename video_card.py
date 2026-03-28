"""
video_card.py — IPL 2026 Match Prediction Card Generator

Generates a professional dark-themed prediction card image (PNG) for a given
IPL match. Designed to be used as a video thumbnail or match-day graphic.

Dependencies: matplotlib, Pillow (PIL) — no external APIs needed.

Usage:
    from video_card import generate_prediction_card

    generate_prediction_card(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        venue="Wankhede Stadium",
        predicted_winner="Mumbai Indians",
        win_prob_team1=0.62,
        win_prob_team2=0.38,
        h2h=(14, 12),          # (team1 wins, team2 wins)
        form_team1="WWLWW",
        form_team2="LWWLW",
        venue_win_pct_team1=58.0,
        venue_win_pct_team2=42.0,
        date="2026-03-28",
    )
    # Saves to: output/prediction_card_Mumbai_Indians_vs_Chennai_Super_Kings_2026-03-28.png
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Colours — IPL dark theme
# ---------------------------------------------------------------------------
BG_DARK = "#0D1117"
BG_CARD = "#161B22"
BG_PANEL = "#21262D"
ACCENT_GOLD = "#FFD700"
ACCENT_TEAL = "#00D4AA"
TEXT_PRIMARY = "#F0F6FC"
TEXT_SECONDARY = "#8B949E"
WIN_GREEN = "#3FB950"
LOSS_RED = "#F85149"
DRAW_GREY = "#484F58"

# Team brand colours (primary)
TEAM_COLORS: dict[str, str] = {
    "Mumbai Indians": "#005DA0",
    "Chennai Super Kings": "#F7D02C",
    "Royal Challengers Bengaluru": "#EC1C24",
    "Royal Challengers Bangalore": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Delhi Capitals": "#0078BC",
    "Sunrisers Hyderabad": "#F7692A",
    "Rajasthan Royals": "#EA1A7F",
    "Punjab Kings": "#D71920",
    "Lucknow Super Giants": "#A72B7F",
    "Gujarat Titans": "#1C2951",
}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _team_color(team: str) -> str:
    return TEAM_COLORS.get(team, ACCENT_TEAL)


def _form_char_color(ch: str) -> str:
    return WIN_GREEN if ch == "W" else (LOSS_RED if ch == "L" else DRAW_GREY)


def generate_prediction_card(
    team1: str,
    team2: str,
    venue: str,
    predicted_winner: str,
    win_prob_team1: float,
    win_prob_team2: float,
    h2h: tuple[int, int] = (0, 0),
    form_team1: str = "WWWWW",
    form_team2: str = "WWWWW",
    venue_win_pct_team1: float = 50.0,
    venue_win_pct_team2: float = 50.0,
    date: str | None = None,
    confidence: float | None = None,
    save_path: str | None = None,
) -> str:
    """
    Generate a match prediction card image.

    Parameters
    ----------
    team1, team2          : Team names
    venue                 : Match venue
    predicted_winner      : Name of predicted winner
    win_prob_team1/2      : Float in [0, 1] — win probabilities
    h2h                   : (team1 wins, team2 wins) in head-to-head
    form_team1/2          : 5-char string of W/L for last 5 matches
    venue_win_pct_team1/2 : Win % at this venue (0–100)
    date                  : Match date string (default: today)
    confidence            : Override confidence % (default: max prob * 100)
    save_path             : Override output file path

    Returns
    -------
    str : Path to the saved PNG image
    """
    date = date or datetime.today().strftime("%Y-%m-%d")
    if confidence is None:
        confidence = max(win_prob_team1, win_prob_team2) * 100

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        t1_slug = team1.replace(" ", "_")
        t2_slug = team2.replace(" ", "_")
        save_path = str(OUTPUT_DIR / f"prediction_card_{t1_slug}_vs_{t2_slug}_{date}.png")

    # -----------------------------------------------------------------------
    # Figure layout  (1200 × 680 px at 100 dpi)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 6.8), dpi=100, facecolor=BG_DARK)

    # Main axes — full canvas
    ax_main = fig.add_axes([0, 0, 1, 1], facecolor=BG_DARK)
    ax_main.set_xlim(0, 12)
    ax_main.set_ylim(0, 6.8)
    ax_main.axis("off")

    # -----------------------------------------------------------------------
    # Background card
    # -----------------------------------------------------------------------
    card = FancyBboxPatch((0.3, 0.3), 11.4, 6.2, boxstyle="round,pad=0.15",
                          facecolor=BG_CARD, edgecolor=ACCENT_GOLD, linewidth=2, zorder=1)
    ax_main.add_patch(card)

    # -----------------------------------------------------------------------
    # Header bar
    # -----------------------------------------------------------------------
    header = FancyBboxPatch((0.3, 5.5), 11.4, 0.95, boxstyle="round,pad=0.05",
                            facecolor=BG_PANEL, edgecolor="none", zorder=2)
    ax_main.add_patch(header)

    ax_main.text(6, 6.18, "🏆  IPL 2026  •  Match Prediction Card",
                 ha="center", va="center", fontsize=16, fontweight="bold",
                 color=ACCENT_GOLD, zorder=3)
    ax_main.text(6, 5.72, f"{team1}  vs  {team2}  —  {date}",
                 ha="center", va="center", fontsize=11, color=TEXT_SECONDARY, zorder=3)

    # -----------------------------------------------------------------------
    # Team colour bars (left & right)
    # -----------------------------------------------------------------------
    t1_bar = FancyBboxPatch((0.3, 0.3), 0.25, 5.2, boxstyle="square,pad=0",
                            facecolor=_team_color(team1), edgecolor="none", zorder=2)
    t2_bar = FancyBboxPatch((11.45, 0.3), 0.25, 5.2, boxstyle="square,pad=0",
                            facecolor=_team_color(team2), edgecolor="none", zorder=2)
    ax_main.add_patch(t1_bar)
    ax_main.add_patch(t2_bar)

    # -----------------------------------------------------------------------
    # VS text
    # -----------------------------------------------------------------------
    ax_main.text(6, 4.55, "VS", ha="center", va="center",
                 fontsize=28, fontweight="black", color=TEXT_SECONDARY, alpha=0.4, zorder=3)

    # -----------------------------------------------------------------------
    # Team names
    # -----------------------------------------------------------------------
    ax_main.text(2.8, 4.55, team1, ha="center", va="center",
                 fontsize=14, fontweight="bold", color=_team_color(team1), zorder=3,
                 wrap=True)
    ax_main.text(9.2, 4.55, team2, ha="center", va="center",
                 fontsize=14, fontweight="bold", color=_team_color(team2), zorder=3,
                 wrap=True)

    # -----------------------------------------------------------------------
    # Venue
    # -----------------------------------------------------------------------
    ax_main.text(6, 3.95, f"📍 {venue}", ha="center", va="center",
                 fontsize=9.5, color=TEXT_SECONDARY, zorder=3)

    # -----------------------------------------------------------------------
    # Win probability bar
    # -----------------------------------------------------------------------
    bar_y, bar_h = 3.35, 0.38
    bar_x0, bar_w = 1.0, 10.0

    # background
    ax_main.add_patch(FancyBboxPatch((bar_x0, bar_y), bar_w, bar_h,
                                     boxstyle="round,pad=0.05",
                                     facecolor=DRAW_GREY, edgecolor="none", zorder=3))
    # team1 fill
    t1_fill = bar_w * win_prob_team1
    ax_main.add_patch(FancyBboxPatch((bar_x0, bar_y), t1_fill, bar_h,
                                     boxstyle="round,pad=0.05",
                                     facecolor=_team_color(team1), edgecolor="none",
                                     alpha=0.85, zorder=4))
    # team2 fill (right side)
    t2_fill = bar_w * win_prob_team2
    ax_main.add_patch(FancyBboxPatch((bar_x0 + bar_w - t2_fill, bar_y), t2_fill, bar_h,
                                     boxstyle="round,pad=0.05",
                                     facecolor=_team_color(team2), edgecolor="none",
                                     alpha=0.85, zorder=4))

    ax_main.text(bar_x0 + 0.2, bar_y + bar_h / 2,
                 f"{win_prob_team1 * 100:.0f}%",
                 ha="left", va="center", fontsize=11, fontweight="bold",
                 color=TEXT_PRIMARY, zorder=5)
    ax_main.text(bar_x0 + bar_w - 0.2, bar_y + bar_h / 2,
                 f"{win_prob_team2 * 100:.0f}%",
                 ha="right", va="center", fontsize=11, fontweight="bold",
                 color=TEXT_PRIMARY, zorder=5)

    # Win probability label
    ax_main.text(6, bar_y + bar_h + 0.15, "Win Probability",
                 ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY, zorder=5)

    # -----------------------------------------------------------------------
    # Stats panels — Form, H2H, Venue
    # -----------------------------------------------------------------------
    panels_y = 1.55
    panel_h = 1.55
    panel_configs = [
        (1.0, 3.0, "Recent Form (Last 5)"),
        (4.5, 3.0, "Head-to-Head"),
        (8.0, 3.0, "Venue Win %"),
    ]

    for (px, pw, title) in panel_configs:
        ax_main.add_patch(FancyBboxPatch((px, panels_y), pw, panel_h,
                                         boxstyle="round,pad=0.08",
                                         facecolor=BG_PANEL, edgecolor=DRAW_GREY,
                                         linewidth=0.8, zorder=3))
        ax_main.text(px + pw / 2, panels_y + panel_h - 0.15, title,
                     ha="center", va="top", fontsize=8.5, color=TEXT_SECONDARY, zorder=4)

    # --- Form panel ---
    px, pw = 1.0, 3.0
    circle_r = 0.14
    for row_idx, (team_label, form) in enumerate([(team1, form_team1), (team2, form_team2)]):
        row_y = panels_y + 0.9 - row_idx * 0.45
        short_name = team_label.split()[-1][:3].upper()
        ax_main.text(px + 0.25, row_y, short_name,
                     ha="center", va="center", fontsize=7.5,
                     color=TEXT_SECONDARY, zorder=4)
        for ci, ch in enumerate(form[-5:]):
            cx = px + 0.65 + ci * 0.42
            circle = plt.Circle((cx, row_y), circle_r,
                                 color=_form_char_color(ch), zorder=4)
            ax_main.add_patch(circle)
            ax_main.text(cx, row_y, ch, ha="center", va="center",
                         fontsize=7, fontweight="bold", color=TEXT_PRIMARY, zorder=5)

    # --- H2H panel ---
    px, pw = 4.5, 3.0
    h2h_t1, h2h_t2 = h2h
    h2h_total = h2h_t1 + h2h_t2 or 1
    ax_main.text(px + pw / 2, panels_y + 0.9, f"{h2h_t1}  —  {h2h_t2}",
                 ha="center", va="center", fontsize=22, fontweight="black",
                 color=TEXT_PRIMARY, zorder=4)
    ax_main.text(px + 0.45, panels_y + 0.42,
                 team1.split()[-1][:6], ha="center", va="center",
                 fontsize=8, color=_team_color(team1), zorder=4)
    ax_main.text(px + pw - 0.45, panels_y + 0.42,
                 team2.split()[-1][:6], ha="center", va="center",
                 fontsize=8, color=_team_color(team2), zorder=4)
    ax_main.text(px + pw / 2, panels_y + 0.42,
                 f"({h2h_total} matches)", ha="center", va="center",
                 fontsize=7.5, color=TEXT_SECONDARY, zorder=4)

    # --- Venue Win % panel ---
    px, pw = 8.0, 3.0
    ax_axes = fig.add_axes([
        (px + 0.3) / 12, (panels_y + 0.15) / 6.8,
        (pw - 0.6) / 12, 0.8 / 6.8
    ], facecolor=BG_PANEL)
    ax_axes.set_facecolor(BG_PANEL)
    bars = ax_axes.barh(
        [team2.split()[-1][:8], team1.split()[-1][:8]],
        [venue_win_pct_team2, venue_win_pct_team1],
        color=[_team_color(team2), _team_color(team1)],
        height=0.45, alpha=0.85,
    )
    for bar, val in zip(bars, [venue_win_pct_team2, venue_win_pct_team1]):
        ax_axes.text(min(val + 2, 95), bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f}%", va="center", fontsize=8,
                     color=TEXT_PRIMARY, fontweight="bold")
    ax_axes.set_xlim(0, 105)
    ax_axes.set_xticks([])
    ax_axes.tick_params(colors=TEXT_SECONDARY, labelsize=7.5)
    for spine in ax_axes.spines.values():
        spine.set_visible(False)
    ax_axes.tick_params(axis="y", length=0)
    plt.setp(ax_axes.get_yticklabels(), color=TEXT_SECONDARY)

    # -----------------------------------------------------------------------
    # Prediction badge
    # -----------------------------------------------------------------------
    badge_y = 0.45
    badge = FancyBboxPatch((2.5, badge_y), 7.0, 0.85,
                            boxstyle="round,pad=0.1",
                            facecolor=_team_color(predicted_winner),
                            edgecolor=ACCENT_GOLD, linewidth=2, zorder=5, alpha=0.92)
    ax_main.add_patch(badge)
    ax_main.text(6, badge_y + 0.42,
                 f"🏆  Prediction: {predicted_winner} wins!  ({confidence:.0f}% confidence)",
                 ha="center", va="center", fontsize=12, fontweight="bold",
                 color=TEXT_PRIMARY, zorder=6)

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    ax_main.text(6, 0.22, "IPL 2026 • Ensemble ML Model (LightGBM + CatBoost + XGBoost)",
                 ha="center", va="center", fontsize=7.5,
                 color=TEXT_SECONDARY, alpha=0.7, zorder=3)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    plt.savefig(save_path, dpi=100, bbox_inches="tight",
                facecolor=BG_DARK, edgecolor="none")
    plt.close(fig)
    print(f"Prediction card saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate IPL prediction card")
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--winner", required=True, dest="predicted_winner")
    parser.add_argument("--prob1", type=float, required=True)
    parser.add_argument("--prob2", type=float, required=True)
    parser.add_argument("--h2h", default="0,0",
                        help="Comma-separated team1_wins,team2_wins")
    parser.add_argument("--form1", default="WWWWW")
    parser.add_argument("--form2", default="WWWWW")
    parser.add_argument("--venue-pct1", type=float, default=50.0)
    parser.add_argument("--venue-pct2", type=float, default=50.0)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    h2h_vals = tuple(int(x) for x in args.h2h.split(","))
    generate_prediction_card(
        team1=args.team1,
        team2=args.team2,
        venue=args.venue,
        predicted_winner=args.predicted_winner,
        win_prob_team1=args.prob1,
        win_prob_team2=args.prob2,
        h2h=h2h_vals,
        form_team1=args.form1,
        form_team2=args.form2,
        venue_win_pct_team1=args.venue_pct1,
        venue_win_pct_team2=args.venue_pct2,
        date=args.date,
    )
