"""
video_card_demo.py — Run this to generate a sample prediction card with demo data.

No model or dataset needed. Just run:
    python video_card_demo.py

The card is saved to: output/prediction_card_Mumbai_Indians_vs_Chennai_Super_Kings_<date>.png
"""

from video_card import generate_prediction_card
from datetime import date

saved = generate_prediction_card(
    team1="Mumbai Indians",
    team2="Chennai Super Kings",
    venue="Wankhede Stadium, Mumbai",
    predicted_winner="Mumbai Indians",
    win_prob_team1=0.63,
    win_prob_team2=0.37,
    h2h=(18, 14),
    form_team1="WWLWW",
    form_team2="WLWLW",
    venue_win_pct_team1=61.0,
    venue_win_pct_team2=39.0,
    date=str(date.today()),
    confidence=63.0,
)

print(f"\nOpen the card: {saved}")
