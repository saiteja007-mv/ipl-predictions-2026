"""
app.py — IPL 2026 Predictions Streamlit Web App.

4 pages:
  1. Predict Match
  2. Season Tracker
  3. Log Match Result
  4. Stats Explorer
"""

import os
import io
import tempfile
import uuid
import datetime
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IPL 2026 Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark cricket theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ---- Global ---- */
    html, body, [data-testid="stApp"] {
        background-color: #0D1117;
        color: #E6EDF3;
        font-family: 'Segoe UI', sans-serif;
    }
    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    [data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }
    /* ---- Metric cards ---- */
    [data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* ---- Selectbox / Input ---- */
    .stSelectbox > div > div, .stTextArea textarea, .stTextInput input {
        background-color: #21262D !important;
        color: #E6EDF3 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
    }
    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2EA043);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2EA043, #3FB950);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(46,160,67,0.4);
    }
    /* ---- Expanders ---- */
    .streamlit-expanderHeader {
        background-color: #21262D !important;
        border-radius: 6px !important;
        color: #E6EDF3 !important;
    }
    /* ---- Dataframe ---- */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    /* ---- Divider ---- */
    hr { border-color: #30363D; }
    /* ---- Prediction badge ---- */
    .prediction-badge {
        background: linear-gradient(135deg, #1C2951, #0D1117);
        border: 2px solid #FFD700;
        border-radius: 12px;
        padding: 20px 30px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-badge h1 { color: #FFD700; font-size: 2rem; margin: 0; }
    .prediction-badge p { color: #8B949E; margin: 4px 0 0 0; }
    /* ---- Form pill ---- */
    .form-pill-W {
        display: inline-block; width: 28px; height: 28px; border-radius: 50%;
        background: #238636; color: white; text-align: center;
        line-height: 28px; font-weight: bold; margin: 2px;
    }
    .form-pill-L {
        display: inline-block; width: 28px; height: 28px; border-radius: 50%;
        background: #B22222; color: white; text-align: center;
        line-height: 28px; font-weight: bold; margin: 2px;
    }
    .form-pill-? {
        display: inline-block; width: 28px; height: 28px; border-radius: 50%;
        background: #30363D; color: #8B949E; text-align: center;
        line-height: 28px; font-weight: bold; margin: 2px;
    }
    /* ---- Section header ---- */
    .section-header {
        background-color: #161B22;
        border-left: 4px solid #FFD700;
        padding: 8px 16px;
        border-radius: 0 6px 6px 0;
        margin: 1rem 0 0.5rem 0;
        font-weight: 700;
        font-size: 1.1rem;
        color: #E6EDF3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Imports from predictor (after CSS, before any logic)
# ---------------------------------------------------------------------------

try:
    import predictor as pred
    PREDICTOR_OK = True
except ImportError as _e:
    st.error(f"Failed to import predictor.py: {_e}")
    PREDICTOR_OK = False

# XI extractor — optional; gracefully degrade if OCR deps not installed
try:
    from xi_extractor import extract_xi_from_image
    XI_EXTRACTOR_OK = True
except ImportError:
    XI_EXTRACTOR_OK = False

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    """Load model + feature metadata (cached for the session)."""
    return pred.load_model()


@st.cache_resource(show_spinner="Loading Elo ratings…")
def _load_elo():
    """Load Elo ratings (cached for the session)."""
    return pred.load_elo_ratings()


@st.cache_data(show_spinner="Loading datasets…")
def _load_data():
    """Load core DataFrames (cached for the session)."""
    return pred.load_data()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "🏏 Predict Match": "predict",
    "📊 Season Tracker": "tracker",
    "📝 Log Match Result": "log",
    "📈 Stats Explorer": "stats",
}

with st.sidebar:
    st.markdown("## 🏏 IPL 2026 Predictor")
    st.markdown("---")
    page_label = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]
    st.markdown("---")
    st.caption("Powered by Ensemble ML\n(LightGBM + CatBoost + XGBoost)")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def form_pills_html(form_str: str) -> str:
    """Render a form string (e.g. 'WWLWW') as colored HTML pills."""
    html = ""
    for ch in form_str:
        cls = f"form-pill-{ch}" if ch in ("W", "L") else "form-pill-?"
        html += f'<span class="{cls}">{ch}</span>'
    return html


def team_badge(team_name: str) -> str:
    """Return a colored emoji+abbr badge for a team."""
    abbr = pred.TEAM_ABBR.get(team_name, team_name[:3].upper())
    color = pred.TEAM_COLORS.get(team_name, "#444")
    return f'<span style="background:{color};color:white;padding:3px 8px;border-radius:4px;font-weight:bold;font-size:0.85rem;">{abbr}</span>'


def prob_bar_html(team1: str, team2: str, p1: float, p2: float) -> str:
    """Render a side-by-side probability bar as HTML."""
    c1 = pred.TEAM_COLORS.get(team1, "#004BA0")
    c2 = pred.TEAM_COLORS.get(team2, "#D1001C")
    pct1 = round(p1 * 100, 1)
    pct2 = round(p2 * 100, 1)
    return f"""
    <div style="margin:1rem 0;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="color:{c1};font-weight:bold;width:200px;text-align:right;">{team1}</span>
        <div style="flex:1;height:32px;background:#21262D;border-radius:6px;overflow:hidden;display:flex;">
          <div style="width:{pct1}%;background:{c1};display:flex;align-items:center;justify-content:flex-end;padding-right:6px;">
            <span style="color:white;font-weight:bold;font-size:0.85rem;">{pct1}%</span>
          </div>
          <div style="width:{pct2}%;background:{c2};display:flex;align-items:center;justify-content:flex-start;padding-left:6px;">
            <span style="color:white;font-weight:bold;font-size:0.85rem;">{pct2}%</span>
          </div>
        </div>
        <span style="color:{c2};font-weight:bold;width:200px;">{team2}</span>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Page 1: Predict Match
# ---------------------------------------------------------------------------


def page_predict() -> None:
    """Render the Predict Match page."""
    st.markdown("# 🏏 Predict Match")
    st.markdown("Fill in the match details and hit **Predict!** to get the model's forecast.")
    st.markdown("---")

    if not PREDICTOR_OK:
        st.error("predictor.py failed to import.")
        return

    # Load resources
    model, feature_columns, team_name_map = _load_model()
    if model is None:
        st.warning("⚠️ **Model not trained yet.** Run `training.ipynb` first.")
        return

    matches, player_match_stats, player_stats = _load_data()
    if matches.empty:
        st.warning("⚠️ **Datasets not found.** Run `training.ipynb` to download.")
        return

    elo_ratings = _load_elo()

    # --- Input form ---
    col1, col2, col3 = st.columns(3)
    with col1:
        team1 = st.selectbox("Team 1", pred.IPL_TEAMS, key="p_team1")
    with col2:
        team2_options = [t for t in pred.IPL_TEAMS if t != team1]
        team2 = st.selectbox("Team 2", team2_options, key="p_team2")
    with col3:
        venue = st.selectbox("Venue", pred.IPL_VENUES, key="p_venue")

    col4, col5 = st.columns(2)
    with col4:
        toss_winner = st.selectbox("Toss Winner", [team1, team2], key="p_toss_winner")
    with col5:
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"], key="p_toss_dec")

    # --- Playing XI (Optional) ---
    st.subheader("📋 Playing XI (Optional — improves prediction accuracy)")

    xi_input_method = st.radio(
        "How to enter Playing XI?",
        ["Skip", "Type names manually", "Upload photo of XI announcement"],
        horizontal=True,
        key="xi_method",
    )

    # Hint when photo upload selected but deps missing
    if xi_input_method == "Upload photo of XI announcement" and not XI_EXTRACTOR_OK:
        st.warning(
            "OCR dependencies not installed. "
            "Run `pip install pytesseract Pillow rapidfuzz` and "
            "`sudo apt-get install tesseract-ocr`, then restart the app."
        )
        xi_input_method = "Skip"

    team1_xi: list[str] = []
    team2_xi: list[str] = []

    if xi_input_method == "Type names manually":
        xi_col1, xi_col2 = st.columns(2)
        with xi_col1:
            t1_text = st.text_area(
                f"{team1} XI (one name per line)", height=200, key="p_xi1_text"
            )
            team1_xi = [n.strip() for n in t1_text.splitlines() if n.strip()]
        with xi_col2:
            t2_text = st.text_area(
                f"{team2} XI (one name per line)", height=200, key="p_xi2_text"
            )
            team2_xi = [n.strip() for n in t2_text.splitlines() if n.strip()]

        if team1_xi:
            st.caption(f"{team1}: {len(team1_xi)} player(s) entered")
        if team2_xi:
            st.caption(f"{team2}: {len(team2_xi)} player(s) entered")

    elif xi_input_method == "Upload photo of XI announcement":
        xi_col1, xi_col2 = st.columns(2)

        with xi_col1:
            st.write(f"**{team1} XI Photo**")
            t1_img = st.file_uploader(
                f"Upload {team1} XI",
                type=["png", "jpg", "jpeg"],
                key="t1_xi_upload",
            )
            if t1_img is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(t1_img.read())
                    tmp_path = tmp.name
                with st.spinner("Extracting players from image…"):
                    try:
                        team1_xi = extract_xi_from_image(tmp_path)
                    except ImportError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"OCR failed: {exc}")
                if team1_xi:
                    st.success(f"Found {len(team1_xi)} player(s)")
                    st.write(team1_xi)
                else:
                    st.warning("Could not extract names — please type manually")

        with xi_col2:
            st.write(f"**{team2} XI Photo**")
            t2_img = st.file_uploader(
                f"Upload {team2} XI",
                type=["png", "jpg", "jpeg"],
                key="t2_xi_upload",
            )
            if t2_img is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(t2_img.read())
                    tmp_path = tmp.name
                with st.spinner("Extracting players from image…"):
                    try:
                        team2_xi = extract_xi_from_image(tmp_path)
                    except ImportError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"OCR failed: {exc}")
                if team2_xi:
                    st.success(f"Found {len(team2_xi)} player(s)")
                    st.write(team2_xi)
                else:
                    st.warning("Could not extract names — please type manually")

    st.markdown("")
    predict_btn = st.button("🔮 Predict!", use_container_width=False)

    if predict_btn:
        with st.spinner("Running prediction…"):
            try:
                result = pred.predict_match(
                    team1=team1,
                    team2=team2,
                    venue=venue,
                    toss_winner=toss_winner,
                    toss_decision=toss_decision,
                    model=model,
                    feature_columns=feature_columns,
                    matches=matches,
                    player_match_stats=player_match_stats,
                    elo_ratings=elo_ratings,
                    team1_xi=team1_xi or None,
                    team2_xi=team2_xi or None,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        winner = result["winner"]
        p1 = result["prob_team1"]
        p2 = result["prob_team2"]
        conf = result["confidence"]
        form1 = result["form_team1"]
        form2 = result["form_team2"]
        h2h = result["h2h"]
        vstats = result["venue_stats"]
        winner_color = pred.TEAM_COLORS.get(winner, "#FFD700")

        # --- Prediction badge ---
        st.markdown(
            f"""
            <div class="prediction-badge" style="border-color:{winner_color};">
              <h1 style="color:{winner_color};">🏆 PREDICTED WINNER: {winner.upper()}</h1>
              <p style="font-size:1.2rem;color:#8B949E;">Confidence: <strong style="color:{winner_color};">{conf*100:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Probability bar ---
        st.markdown(prob_bar_html(team1, team2, p1, p2), unsafe_allow_html=True)

        # --- Stats panels ---
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            with st.expander("📋 Head-to-Head Record", expanded=True):
                t1w, t2w, total = h2h
                h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
                h2h_col1.metric(team1, f"{t1w} wins")
                h2h_col2.metric("Total", total)
                h2h_col3.metric(team2, f"{t2w} wins")

            with st.expander("🏟️ Venue Stats", expanded=True):
                v_col1, v_col2, v_col3 = st.columns(3)
                v_col1.metric("Bat-first win %", f"{vstats['bat_first_pct']}%")
                v_col2.metric(f"{team1} at venue", f"{vstats['team1_pct']}%")
                v_col3.metric(f"{team2} at venue", f"{vstats['team2_pct']}%")

        with exp_col2:
            with st.expander("🔥 Recent Form (last 5)", expanded=True):
                st.markdown(
                    f"**{team1}**: {form_pills_html(form1)}<br>"
                    f"**{team2}**: {form_pills_html(form2)}",
                    unsafe_allow_html=True,
                )

            with st.expander("⚡ Team Strength (Elo)", expanded=True):
                elo_ratings_loaded = _load_elo()
                default_elo = 1500.0
                e1 = elo_ratings_loaded.get(team1, default_elo)
                e2 = elo_ratings_loaded.get(team2, default_elo)
                e_col1, e_col2 = st.columns(2)
                e_col1.metric(f"{team1} Elo", f"{e1:.0f}")
                e_col2.metric(f"{team2} Elo", f"{e2:.0f}")

        # --- Prediction card ---
        st.markdown("---")
        st.markdown("### 🖼️ Prediction Card")
        card_col, dl_col = st.columns([3, 1])

        with card_col:
            with st.spinner("Generating prediction card…"):
                try:
                    from video_card import generate_prediction_card
                    today = datetime.date.today().isoformat()
                    card_path = generate_prediction_card(
                        team1=team1,
                        team2=team2,
                        venue=venue,
                        predicted_winner=winner,
                        win_prob_team1=p1,
                        win_prob_team2=p2,
                        h2h=(h2h[0], h2h[1]),
                        form_team1=form1[:5] if form1 else "?????",
                        form_team2=form2[:5] if form2 else "?????",
                        venue_win_pct_team1=vstats["team1_pct"],
                        venue_win_pct_team2=vstats["team2_pct"],
                        date=today,
                        confidence=conf * 100,
                    )
                    st.image(card_path, use_container_width=True)

                    with dl_col:
                        st.markdown("<br><br><br>", unsafe_allow_html=True)
                        with open(card_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download Card",
                                data=f,
                                file_name=os.path.basename(card_path),
                                mime="image/png",
                                use_container_width=True,
                            )
                except Exception as e:
                    st.warning(f"Card generation failed: {e}")

        # --- Save to tracker ---
        st.markdown("---")
        if st.button("💾 Save Prediction to Season Tracker"):
            tracker = pred.load_season_tracker()
            match_id = f"{team1}_vs_{team2}_{datetime.date.today().isoformat()}_{uuid.uuid4().hex[:6]}"
            tracker = pred.add_prediction_to_tracker(
                tracker=tracker,
                match_id=match_id,
                date=datetime.date.today().isoformat(),
                team1=team1,
                team2=team2,
                venue=venue,
                predicted_winner=winner,
                confidence=conf,
            )
            pred.save_season_tracker(tracker)
            st.success(f"✅ Prediction saved! Match ID: `{match_id}`")


# ---------------------------------------------------------------------------
# Page 2: Season Tracker
# ---------------------------------------------------------------------------


def page_tracker() -> None:
    """Render the Season Tracker page."""
    st.markdown("# 📊 Season Tracker")
    st.markdown("Track all predictions and compare against actual results.")
    st.markdown("---")

    tracker = pred.load_season_tracker()
    predictions = tracker.get("predictions", [])

    correct, total, accuracy = pred.compute_accuracy(tracker)
    pending = len([p for p in predictions if p.get("correct") is None])

    # --- Summary metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Predictions", len(predictions))
    m2.metric("Results Logged", total)
    m3.metric("Correct", f"{correct}/{total}" if total > 0 else "—")
    m4.metric("Accuracy", f"{accuracy}%" if total > 0 else "—")

    if total > 0:
        st.markdown(
            f"""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;
                        padding:16px;margin:1rem 0;text-align:center;">
              <span style="font-size:1.5rem;font-weight:bold;color:#FFD700;">
                {correct}/{total} correct — {accuracy}% accuracy
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if not predictions:
        st.info("No predictions yet. Go to **🏏 Predict Match** to make your first prediction!")
        return

    # --- Data table ---
    st.markdown('<div class="section-header">📋 All Predictions</div>', unsafe_allow_html=True)
    df = pd.DataFrame(predictions)

    display_cols = [c for c in ["date", "team1", "team2", "predicted_winner", "actual_winner", "correct", "confidence"] if c in df.columns]
    display_df = df[display_cols].copy()
    if "correct" in display_df.columns:
        display_df["correct"] = display_df["correct"].map(
            lambda x: "✅" if x is True else ("❌" if x is False else "⏳")
        )
    if "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
        )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Accuracy chart ---
    logged = [p for p in predictions if p.get("correct") is not None]
    if len(logged) >= 2:
        st.markdown("---")
        st.markdown('<div class="section-header">📈 Cumulative Accuracy Over Season</div>', unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        correct_running = []
        acc_running = []
        for i, p in enumerate(logged, 1):
            correct_running.append(sum(1 for lp in logged[:i] if lp["correct"]))
            acc_running.append(correct_running[-1] / i * 100)

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        ax.plot(range(1, len(acc_running) + 1), acc_running, color="#FFD700", linewidth=2.5, marker="o", markersize=5)
        ax.axhline(y=50, color="#30363D", linestyle="--", linewidth=1)
        ax.set_xlabel("Match #", color="#8B949E")
        ax.set_ylabel("Accuracy %", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.spines["bottom"].set_color("#30363D")
        ax.spines["left"].set_color("#30363D")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 105)
        ax.set_title("Running Prediction Accuracy", color="#E6EDF3", fontsize=13)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3: Log Match Result
# ---------------------------------------------------------------------------


def page_log() -> None:
    """Render the Log Match Result page."""
    st.markdown("# 📝 Log Match Result")
    st.markdown("Record the actual outcome of a predicted match.")
    st.markdown("---")

    tracker = pred.load_season_tracker()
    predictions = tracker.get("predictions", [])

    if not predictions:
        st.info("No predictions saved yet. Make a prediction first on the **🏏 Predict Match** page.")
        return

    # Unresolved matches
    unresolved = [p for p in predictions if p.get("correct") is None]
    if not unresolved:
        st.success("✅ All predictions have been logged!")
        return

    match_options = {
        f"{p['team1']} vs {p['team2']} ({p['date']}) — ID: {p['match_id']}": p["match_id"]
        for p in unresolved
    }

    selected_label = st.selectbox("Select Match", list(match_options.keys()))
    match_id = match_options[selected_label]

    selected_pred = next((p for p in predictions if p["match_id"] == match_id), None)
    if selected_pred:
        ic1, ic2 = st.columns(2)
        ic1.info(f"🔮 Predicted: **{selected_pred['predicted_winner']}** ({selected_pred['confidence']*100:.1f}%)")
        ic2.info(f"🏟️ Venue: **{selected_pred.get('venue', '—')}**")

    with st.form("log_result_form"):
        team_options = [selected_pred["team1"], selected_pred["team2"]] if selected_pred else pred.IPL_TEAMS
        actual_winner = st.selectbox("Actual Winner", team_options)
        win_margin = st.text_input("Win Margin (e.g. '5 wickets', '24 runs')", placeholder="Optional")
        player_of_match = st.text_input("Player of the Match", placeholder="Optional")
        also_append_csv = st.checkbox("Also append result to Datasets/matches.csv", value=True)
        submit = st.form_submit_button("✅ Log Result", use_container_width=True)

    if submit:
        # Update tracker
        tracker, found = pred.log_actual_result(tracker, match_id, actual_winner, win_margin, player_of_match)
        if found:
            pred.save_season_tracker(tracker)
            correct_flag = selected_pred["predicted_winner"] == actual_winner if selected_pred else None
            if correct_flag:
                st.success(f"✅ Logged! Prediction was **CORRECT** 🎉")
            else:
                st.error(f"❌ Logged. Prediction was **INCORRECT** — actual winner: **{actual_winner}**")

            correct, total, accuracy = pred.compute_accuracy(tracker)
            st.info(f"📊 Updated accuracy: **{correct}/{total} correct ({accuracy}%)**")

            # Optionally append to CSV
            if also_append_csv and selected_pred:
                try:
                    matches_path = pred.DATASET_DIR / "matches.csv"
                    matches_df = pd.read_csv(matches_path, low_memory=False) if matches_path.exists() else pd.DataFrame()
                    new_row = {
                        "match_id": match_id,
                        "city": pred.VENUE_CITY.get(selected_pred.get("venue", ""), ""),
                        "date": selected_pred.get("date", ""),
                        "season": 2026,
                        "team1": selected_pred["team1"],
                        "team2": selected_pred["team2"],
                        "venue": selected_pred.get("venue", ""),
                        "toss_winner": "",
                        "toss_decision": "",
                        "match_winner": actual_winner,
                        "result": "normal",
                        "win_by_runs": win_margin if "run" in win_margin.lower() else "",
                        "win_by_wickets": win_margin if "wicket" in win_margin.lower() else "",
                        "player_of_match": player_of_match,
                    }
                    new_df = pd.concat([matches_df, pd.DataFrame([new_row])], ignore_index=True)
                    new_df.to_csv(matches_path, index=False)
                    st.success("✅ Appended to Datasets/matches.csv")
                except Exception as e:
                    st.warning(f"Could not append to CSV: {e}")
        else:
            st.error(f"Match ID `{match_id}` not found in tracker.")


# ---------------------------------------------------------------------------
# Page 4: Stats Explorer
# ---------------------------------------------------------------------------


def page_stats() -> None:
    """Render the Stats Explorer page."""
    st.markdown("# 📈 Stats Explorer")
    st.markdown("Explore team performance, venue trends, and player stats.")
    st.markdown("---")

    matches, player_match_stats, player_stats = _load_data()

    if matches.empty:
        st.warning("⚠️ **Datasets not found.** Run `training.ipynb` to download.")
        return

    tab1, tab2, tab3 = st.tabs(["🏏 Team Stats", "🏟️ Venue Stats", "👤 Player Search"])

    # ---- Tab 1: Team Stats ----
    with tab1:
        team_sel = st.selectbox("Select Team", pred.IPL_TEAMS, key="stats_team")
        st.markdown("---")

        all_time = pred.get_team_all_time_stats(team_sel, matches)
        _, form_str = pred.get_team_form(team_sel, matches, n=5)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Matches", all_time["total_matches"])
        col2.metric("Total Wins", all_time["wins"])
        col3.metric("Win Rate", f"{all_time['win_rate']}%")
        col4.metric("Recent Form", form_str or "—")

        st.markdown("")
        st.markdown(f"**Recent Form:** {form_pills_html(form_str)}", unsafe_allow_html=True)

        # Recent form chart (last 10)
        recent_df = pred.get_team_recent_form_series(team_sel, matches, n=10)
        if not recent_df.empty:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Last 10 Matches</div>', unsafe_allow_html=True)

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 2.5))
            fig.patch.set_facecolor("#0D1117")
            ax.set_facecolor("#161B22")
            colors = ["#238636" if r == "W" else "#B22222" for r in recent_df["result"]]
            labels = [f"{r}\nvs {str(opp)[:10]}" for r, opp in zip(recent_df["result"], recent_df["opponent"])]
            bars = ax.bar(range(len(recent_df)), [1] * len(recent_df), color=colors, edgecolor="#30363D", width=0.7)
            ax.set_xticks(range(len(recent_df)))
            ax.set_xticklabels(labels, fontsize=8, color="#E6EDF3")
            ax.set_yticks([])
            ax.spines["bottom"].set_color("#30363D")
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f"{team_sel} — Last 10 Results", color="#E6EDF3", fontsize=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.dataframe(recent_df[["date", "opponent", "result", "venue"]], use_container_width=True, hide_index=True)

        # H2H matrix heatmap
        st.markdown("---")
        st.markdown('<div class="section-header">🔥 Head-to-Head Heatmap (vs all IPL teams)</div>', unsafe_allow_html=True)
        with st.spinner("Computing H2H matrix…"):
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                h2h_teams = pred.IPL_TEAMS
                matrix = pred.get_h2h_matrix(matches, h2h_teams)

                fig, ax = plt.subplots(figsize=(10, 7))
                fig.patch.set_facecolor("#0D1117")
                ax.set_facecolor("#161B22")
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="YlOrRd",
                    linewidths=0.5,
                    linecolor="#30363D",
                    ax=ax,
                    cbar_kws={"shrink": 0.8},
                )
                ax.set_title("Head-to-Head Wins Matrix", color="#E6EDF3", fontsize=13, pad=12)
                ax.tick_params(colors="#E6EDF3", labelsize=8)
                ax.set_xlabel("Opponent", color="#8B949E")
                ax.set_ylabel("Team", color="#8B949E")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.warning(f"H2H heatmap failed: {e}")

    # ---- Tab 2: Venue Stats ----
    with tab2:
        venue_sel = st.selectbox("Select Venue", pred.IPL_VENUES, key="stats_venue")
        st.markdown("---")

        if "venue" in matches.columns:
            venue_m = matches[matches["venue"].str.contains(venue_sel.split(",")[0], na=False, case=False)]
            total_v = len(venue_m)

            if total_v == 0:
                st.info(f"No match data found for **{venue_sel}**.")
            else:
                wc = "match_winner" if "match_winner" in matches.columns else "winner"

                # Bat-first stats
                bat_first = venue_m[venue_m.get("toss_decision", pd.Series()) == "bat"]
                field_first = venue_m[venue_m.get("toss_decision", pd.Series()) == "field"]

                toss_col = "toss_winner"
                bf_wins = (bat_first[wc] == bat_first[toss_col]).sum() if len(bat_first) > 0 else 0
                ff_wins = (field_first[wc] == field_first[toss_col]).sum() if len(field_first) > 0 else 0

                bf_pct = round(bf_wins / len(bat_first) * 100, 1) if len(bat_first) > 0 else 0
                ff_pct = round(ff_wins / len(field_first) * 100, 1) if len(field_first) > 0 else 0

                vc1, vc2, vc3 = st.columns(3)
                vc1.metric("Total Matches at Venue", total_v)
                vc2.metric("Bat-first Win %", f"{bf_pct}%")
                vc3.metric("Chase Win %", f"{ff_pct}%")

                # Top teams at venue
                st.markdown("---")
                st.markdown('<div class="section-header">🏆 Top Teams at This Venue</div>', unsafe_allow_html=True)
                team_stats_venue = []
                for team in pred.IPL_TEAMS:
                    t_m = venue_m[(venue_m.get("team1") == team) | (venue_m.get("team2") == team)]
                    if len(t_m) > 0:
                        wins = (t_m[wc] == team).sum()
                        team_stats_venue.append({
                            "Team": team,
                            "Matches": len(t_m),
                            "Wins": int(wins),
                            "Win %": f"{round(wins/len(t_m)*100,1)}%",
                        })

                if team_stats_venue:
                    tv_df = pd.DataFrame(team_stats_venue).sort_values("Wins", ascending=False)
                    st.dataframe(tv_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Venue column not found in matches dataset.")

    # ---- Tab 3: Player Search ----
    with tab3:
        query = st.text_input("🔍 Search Player", placeholder="Type a player name…")
        if query:
            results = pred.search_player_stats(query, player_stats)
            if results.empty:
                st.info(f"No players found matching '{query}'.")
            else:
                st.markdown(f"**{len(results)} players found:**")
                st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.info("Enter a player name to search career statistics.")

        # Per-match stats search
        if not player_match_stats.empty:
            st.markdown("---")
            st.markdown('<div class="section-header">📋 Per-Match Performance Lookup</div>', unsafe_allow_html=True)
            pm_query = st.text_input("Search by player name (match-by-match)", key="pm_search",
                                     placeholder="e.g. Virat Kohli")
            if pm_query:
                pm_col = next((c for c in player_match_stats.columns if "player" in c.lower()), None)
                if pm_col:
                    pm_results = player_match_stats[
                        player_match_stats[pm_col].astype(str).str.contains(pm_query, case=False, na=False)
                    ].head(30)
                    if not pm_results.empty:
                        st.dataframe(pm_results, use_container_width=True, hide_index=True)
                    else:
                        st.info(f"No match data found for '{pm_query}'.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "predict":
    page_predict()
elif page == "tracker":
    page_tracker()
elif page == "log":
    page_log()
elif page == "stats":
    page_stats()
