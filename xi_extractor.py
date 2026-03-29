"""
xi_extractor.py — Playing XI Extraction from Images (OCR + LLM).

Extracts cricket player names from a screenshot or photo of a playing XI
announcement (IPL app, Twitter, TV graphic, etc.) using:

  1. Tesseract OCR  — extracts raw text from the image
  2. Ollama LLM     — (optional, local) parses player names from OCR text
  3. Regex + fuzzy  — fallback when Ollama is unavailable

System requirement (one-time):
    sudo apt-get install tesseract-ocr     # Ubuntu / Debian
    brew install tesseract                 # macOS

Python requirements (already in requirements.txt):
    pytesseract>=0.3
    Pillow>=10.0
    rapidfuzz>=3.0
    requests>=2.31
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    # Auto-detect Tesseract on Windows if not already in PATH
    if os.name == "nt":
        import shutil
        if not shutil.which("tesseract"):
            _win_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            ]
            for _p in _win_paths:
                if os.path.isfile(_p):
                    pytesseract.pytesseract.tesseract_cmd = _p
                    break

    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

try:
    from rapidfuzz import fuzz, process as rfuzz_process

    RAPIDFUZZ_OK = True
except ImportError:
    RAPIDFUZZ_OK = False

try:
    import requests as _requests

    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "Datasets"

# Known player list — populated lazily from player_stats(_enhanced).csv
_KNOWN_PLAYERS: list[str] = []


def _load_known_players() -> list[str]:
    """Load known IPL player names from dataset files (lazily cached)."""
    global _KNOWN_PLAYERS
    if _KNOWN_PLAYERS:
        return _KNOWN_PLAYERS

    for fname in ("player_stats_enhanced.csv", "player_stats.csv"):
        fpath = DATASET_DIR / fname
        if not fpath.exists():
            continue
        try:
            import pandas as pd

            df = pd.read_csv(fpath, low_memory=False)
            name_col = next(
                (c for c in df.columns if c.lower() in ("player", "name", "player_name")),
                None,
            )
            if name_col:
                _KNOWN_PLAYERS = df[name_col].dropna().astype(str).unique().tolist()
                return _KNOWN_PLAYERS
        except Exception:
            continue

    return _KNOWN_PLAYERS


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def ocr_image(image_path: str) -> str:
    """Run Tesseract OCR on an image and return the extracted text.

    Pre-processes the image (greyscale + contrast boost) to improve accuracy
    on low-quality screenshots.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Raw OCR text string.  Empty string if Tesseract is unavailable or
        the image cannot be opened.

    Raises:
        ImportError: If ``pytesseract`` or ``Pillow`` are not installed.
    """
    if not TESSERACT_OK:
        raise ImportError(
            "pytesseract and Pillow are required for OCR.\n"
            "Install with: pip install pytesseract Pillow\n"
            "Also install the Tesseract binary: sudo apt-get install tesseract-ocr"
        )

    try:
        img = Image.open(image_path)
    except Exception as exc:
        print(f"Could not open image {image_path}: {exc}")
        return ""

    # Pre-process: greyscale → sharpen → boost contrast
    img = img.convert("L")  # greyscale
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Custom Tesseract config: treat as uniform block, best accuracy mode
    custom_config = r"--oem 3 --psm 6"

    try:
        text = pytesseract.image_to_string(img, config=custom_config)
    except Exception as exc:
        print(f"Tesseract OCR failed: {exc}")
        return ""

    return text.strip()


# ---------------------------------------------------------------------------
# Ollama LLM parsing (optional)
# ---------------------------------------------------------------------------

def _resolve_ollama_url() -> str:
    """Build a usable Ollama API base URL from OLLAMA_HOST env var."""
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    if not raw:
        return "http://127.0.0.1:11434"
    # Already a full URL
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.rstrip("/")
    # Bind address like "0.0.0.0" or "0.0.0.0:11434" — connect via 127.0.0.1
    host, _, port = raw.partition(":")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = port or "11434"
    return f"http://{host}:{port}"


_OLLAMA_BASE_URL = _resolve_ollama_url()
_OLLAMA_TIMEOUT = 30  # seconds
_OLLAMA_VISION_TIMEOUT = 120  # seconds (large images / slow models)


def _is_ollama_running() -> bool:
    """Check if Ollama API server is reachable on localhost."""
    if not REQUESTS_OK:
        return False
    try:
        resp = _requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def is_ollama_running() -> bool:
    """Public alias: True if the Ollama HTTP API responds (``/api/tags``)."""
    return _is_ollama_running()


def get_ollama_models() -> list[str]:
    """Return installed Ollama model names (e.g. ``llama3.2:latest``, ``llava:latest``).

    Returns an empty list if Ollama is unreachable or ``requests`` is unavailable.
    """
    if not REQUESTS_OK or not _is_ollama_running():
        return []
    try:
        resp = _requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        names = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        return sorted(names)
    except Exception as exc:
        print(f"Could not list Ollama models: {exc}")
        return []


def _clean_name_lines(response_text: str) -> list[str]:
    """Turn model output into a list of player-name lines."""
    if not response_text or not response_text.strip():
        return []
    raw_names = [line.strip() for line in response_text.splitlines() if line.strip()]
    # Strip leading "1." / "1)" numbering
    stripped = []
    for n in raw_names:
        n2 = re.sub(r"^\s*\d+[\).\s]+", "", n).strip()
        if n2:
            stripped.append(n2)
    cleaned = [
        n
        for n in stripped
        if not any(
            kw in n.lower()
            for kw in (
                "here are",
                "here is",
                "following",
                "the players",
                "team:",
                "xi:",
                "playing eleven",
            )
        )
        and len(n) > 2
        and len(n) < 60
    ]
    return cleaned[:11]


def parse_names_with_ollama(text: str, model: str = "llama3.2") -> list[str]:
    """Use a local Ollama LLM to extract player names from OCR text.

    Sends the OCR text to the Ollama ``/api/generate`` endpoint with a prompt
    asking for exactly the 11 player names, one per line.

    Args:
        text: Raw OCR-extracted text from the playing XI image.
        model: Ollama model name to use (default: ``llama3.2``).

    Returns:
        List of player name strings extracted by the LLM.
        Returns empty list if Ollama is not running or the request fails.
    """
    if not REQUESTS_OK:
        return []

    if not _is_ollama_running():
        return []

    prompt = (
        "The following text was extracted from a cricket playing XI announcement image "
        "using OCR. It may contain noise, typos, or formatting artifacts.\n\n"
        f"OCR TEXT:\n{text}\n\n"
        "Task: Extract the 11 cricket player names from this text. "
        "Return ONLY the player names, one per line, with no numbering, bullets, "
        "or extra text. If you cannot identify 11 distinct names, return as many "
        "as you can identify."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256},
    }

    try:
        resp = _requests.post(
            f"{_OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=_OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        response_text = result.get("response", "")
    except Exception as exc:
        print(f"Ollama request failed: {exc}")
        return []

    return _clean_name_lines(response_text)


def extract_names_ollama_vision(image_path: str, model: str) -> list[str]:
    """Use an Ollama **vision** model to read player names directly from the image.

    Calls ``POST /api/chat`` with a base64-encoded image. If the model does not
    support images, Ollama returns an error and this function returns [].

    Args:
        image_path: Path to PNG/JPG/JPEG on disk.
        model: Full Ollama model name (e.g. ``llava:latest``, ``moondream:latest``).

    Returns:
        Parsed player names, or empty list on failure.
    """
    if not REQUESTS_OK or not _is_ollama_running():
        return []

    import base64

    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as exc:
        print(f"Could not read image for vision: {exc}")
        return []

    prompt = (
        "This image is a cricket IPL playing XI (line-up) announcement, app screenshot, "
        "or TV graphic. Extract every player name you can read. "
        "Return ONLY one player name per line, no numbers, bullets, or commentary. "
        "Aim for up to 11 names; if fewer are visible, return those you see."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }

    try:
        resp = _requests.post(
            f"{_OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=_OLLAMA_VISION_TIMEOUT,
        )
        if resp.status_code >= 400:
            print(f"Ollama vision HTTP {resp.status_code}: {resp.text[:300]}")
            return []
        data = resp.json()
        msg = data.get("message") or {}
        response_text = msg.get("content") or ""
    except Exception as exc:
        print(f"Ollama vision request failed: {exc}")
        return []

    return _clean_name_lines(response_text)


# ---------------------------------------------------------------------------
# Regex + fuzzy fallback
# ---------------------------------------------------------------------------

# Common cricket name patterns: "J Smith", "AB de Villiers", "MS Dhoni", etc.
_NAME_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b"  # "Virat Kohli"
    r"|\b([A-Z]{1,3}\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"  # "MS Dhoni", "AB de Villiers"
    r"|\b([A-Z][a-z]+\s+[A-Z]{1,3}\.?\s+[A-Z][a-z]+)\b",  # "Jasprit Bumrah" variants
    re.MULTILINE,
)


def parse_names_with_regex(text: str, known_players: list[str]) -> list[str]:
    """Extract player names from OCR text using regex + fuzzy matching.

    Fallback strategy when Ollama is unavailable.  Applies a regex heuristic
    for capitalized name patterns, then fuzzy-matches against the known player
    database to filter out false positives.

    Args:
        text: Raw OCR-extracted text.
        known_players: List of canonical player names from the database.

    Returns:
        List of candidate player name strings (not yet canonicalized).
    """
    if not text.strip():
        return []

    candidates: list[str] = []

    # Strategy 1: regex pattern matching on name-like tokens
    for match in _NAME_PATTERN.finditer(text):
        name = next((g for g in match.groups() if g), "").strip()
        if name and 3 <= len(name) <= 50:
            candidates.append(name)

    # Strategy 2: line-by-line heuristic
    # Lines with 2–4 words, each starting with a capital, are likely names
    for line in text.splitlines():
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 5:
            if all(w[0].isupper() for w in words if w):
                # Filter out obvious non-names
                if not any(
                    kw.lower() in line.lower()
                    for kw in (
                        "Playing", "XI", "Squad", "Team", "India", "IPL",
                        "Match", "Season", "Captain", "Wicket",
                    )
                ):
                    candidates.append(line)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        c_norm = c.strip().lower()
        if c_norm not in seen:
            seen.add(c_norm)
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# Fuzzy matching against known player DB
# ---------------------------------------------------------------------------


def fuzzy_match_players(
    extracted_names: list[str],
    known_players: list[str],
    threshold: float = 80.0,
) -> list[str]:
    """Fuzzy-match extracted names against known IPL player names.

    Uses ``rapidfuzz`` token_sort_ratio for robust matching of name variants
    (e.g. "V Kohli" → "Virat Kohli", "Dhoni MS" → "MS Dhoni").

    Args:
        extracted_names: Raw names extracted by OCR + LLM/regex.
        known_players: Canonical player names from the database.
        threshold: Minimum similarity score (0–100) to accept a match.

    Returns:
        List of canonical player names for matches above ``threshold``.
        Preserves order and deduplicates.
    """
    if not extracted_names or not known_players:
        return extracted_names  # return as-is if no DB to match against

    if not RAPIDFUZZ_OK:
        # Simple substring fallback
        matched: list[str] = []
        known_lower = {p.lower(): p for p in known_players}
        for name in extracted_names:
            name_l = name.lower()
            # Exact lower-case match
            if name_l in known_lower:
                matched.append(known_lower[name_l])
            else:
                # Partial: any known name that contains this name as substring
                found = next(
                    (canonical for canon_l, canonical in known_lower.items() if name_l in canon_l),
                    name,
                )
                matched.append(found)
        return _deduplicate(matched)

    matched: list[str] = []
    for name in extracted_names:
        result = rfuzz_process.extractOne(
            name,
            known_players,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result:
            matched.append(result[0])
        else:
            # Try a lower threshold one more time
            result2 = rfuzz_process.extractOne(
                name,
                known_players,
                scorer=fuzz.partial_ratio,
                score_cutoff=max(threshold - 15, 60.0),
            )
            if result2:
                matched.append(result2[0])

    return _deduplicate(matched)


def _deduplicate(names: list[str]) -> list[str]:
    """Return names with duplicates removed (case-insensitive), preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        key = n.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(n)
    return result


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_xi_from_image(image_path: str, ollama_model: str | None = None) -> list[str]:
    """Extract playing XI player names from an image.

    Pipeline (when ``ollama_model`` is set and Ollama is running):
    1. Try **vision**: ``extract_names_ollama_vision`` (best for screenshots).
    2. If that returns nothing, **OCR** + same model via ``/api/generate`` (text).
    3. Regex + fuzzy fallback.

    Pipeline (default, ``ollama_model`` is None):
    1. Tesseract OCR → text.
    2. If Ollama is up, default text model parses names; else regex/heuristic.
    3. Fuzzy-match to the IPL player list.

    Args:
        image_path: Path to the uploaded image (PNG/JPG/JPEG).
        ollama_model: Optional Ollama model name from the UI (e.g. ``llava:latest``).
            When set, vision is attempted first; text-only models still work via OCR.

    Returns:
        List of up to 11 canonical player name strings.

    Raises:
        ImportError: If pytesseract/Pillow are not installed **and** no usable
            Ollama vision path produced a result (same as before for OCR-only runs).
    """
    known = _load_known_players()

    def _finalize(extracted: list[str]) -> list[str]:
        if not extracted:
            return []
        print(f"Extracted {len(extracted)} raw names: {extracted}")
        if known:
            matched = fuzzy_match_players(extracted, known)
        else:
            matched = extracted
        final = matched[:11]
        print(f"Final matched names ({len(final)}): {final}")
        return final

    # --- User selected an Ollama model (Streamlit): vision → OCR+LLM → regex ---
    if ollama_model and _is_ollama_running():
        print(f"Using Ollama model: {ollama_model}")
        extracted = extract_names_ollama_vision(image_path, ollama_model)
        if extracted:
            return _finalize(extracted)

        raw_text = ""
        if TESSERACT_OK:
            raw_text = ocr_image(image_path)
        else:
            print("Vision failed and Tesseract not available — cannot OCR.")
            return []

        if not raw_text.strip():
            print("OCR returned no text after empty vision response.")
            return []

        print(f"Vision empty; OCR fallback ({len(raw_text)} chars):\n{raw_text[:400]}\n…")
        extracted = parse_names_with_ollama(raw_text, model=ollama_model)
        if not extracted:
            extracted = parse_names_with_regex(raw_text, known)
        return _finalize(extracted)

    # --- Default: OCR required ---
    if not TESSERACT_OK:
        raise ImportError(
            "pytesseract and Pillow are required.\n"
            "  pip install pytesseract Pillow\n"
            "  sudo apt-get install tesseract-ocr  # (or brew install tesseract on macOS)"
        )

    raw_text = ocr_image(image_path)
    if not raw_text.strip():
        print("OCR returned no text.")
        return []

    print(f"OCR text ({len(raw_text)} chars):\n{raw_text[:400]}\n…")

    if _is_ollama_running():
        print("Ollama detected — using LLM for name extraction.")
        extracted = parse_names_with_ollama(raw_text)
        if not extracted:
            print("Ollama returned no names — falling back to regex.")
            extracted = parse_names_with_regex(raw_text, known)
    else:
        print("Ollama not available — using regex/heuristic extraction.")
        extracted = parse_names_with_regex(raw_text, known)

    if not extracted:
        print("No names extracted from OCR text.")
        return []

    return _finalize(extracted)
