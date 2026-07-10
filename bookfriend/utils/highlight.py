# =========================
# FUNCTION: highlight_keywords
# =========================
import os
import re


# Attempt to import colorama so colored output works on Windows.
# If colorama isn't installed, the script will still run but without colors.
try:
    from colorama import Fore, Style, init as colorama_init
except Exception:
    Fore = None
    Style = None

# Initialize colorama if available so colors reset properly on Windows terminals.
# If colorama not present, this call will be skipped (we check above).
if Fore and Style:
    colorama_init(autoreset=True)


# The relative path to the folder that contains your chapter text files.
# It is built relative to this script's location so the script can be run from anywhere.
CHAPTERS_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # go one level up (from utils/ → bookfriend/)
    "chapters"
)


# How many characters before/after the match to show in the snippet preview.
# This is an approximate context window to help you understand where the match sits.
CONTEXT_CHARS = 40

# List of ANSI color names we'll cycle through for keywords (if colorama available).
# The mapping below maps these names to actual Fore.* color codes.
_COLOR_LIST = ["RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN"]



# -----------------------
# Helper: COLOR_MAP — map friendly names to colorama Fore constants (or empty string fallback)
# -----------------------
if Fore:
    COLOR_MAP = {
        "RED": Fore.RED, "GREEN": Fore.GREEN, "YELLOW": Fore.YELLOW,
        "BLUE": Fore.BLUE, "MAGENTA": Fore.MAGENTA, "CYAN": Fore.CYAN
    }
else:
    # Fallback: if colorama isn't installed, use empty strings so printed text is plain.
    COLOR_MAP = {name: "" for name in _COLOR_LIST}
    # Provide a minimal Style-like object so Style.RESET_ALL usage won't crash.
    class _DummyStyle:
        RESET_ALL = ""
    Style = _DummyStyle()


def build_keyword_color_map(keywords):
    """
    Assign a color to each keyword, cycling through available colors.
    Returns a dict {keyword: color_code_or_empty_string}.
    Why: So the same keyword is displayed consistently in the same color everywhere.
    """
    mapping = {}
    color_names = list(COLOR_MAP.keys())
    for i, kw in enumerate(keywords):
        # choose color by cycling index modulo number of available colors
        color_name = color_names[i % len(color_names)]
        mapping[kw] = COLOR_MAP[color_name]
    return mapping


def highlight_sentence_with_colors(sentence, matches, keywords, kw_color_map, case_sensitive=False):
    """
    Highlight keywords in 'sentence' using per-keyword colors.
    - Exact matches get their assigned color.
    - Fuzzy matches highlight the whole sentence in GREEN.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    result = sentence

    # If any fuzzy match exists, just make the whole sentence green
    if any(m.is_fuzzy for m in matches):
        return f"{COLOR_MAP['GREEN']}{sentence}{Style.RESET_ALL}"

    # Otherwise, highlight exact matches
    for m in matches:
        color = kw_color_map.get(m.keyword, "")
        reset = Style.RESET_ALL if Style else ""
        pattern = r'\b' + re.escape(m.keyword) + r'\b'
        result = re.sub(pattern, lambda mt: f"{color}{mt.group(0)}{reset}", result, flags=flags)

    return result
