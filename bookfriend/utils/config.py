# =========================
# SETTINGS — Change here to switch modes
# =========================

CASE_SENSITIVE_MODE = False  # A global setting:
                             #   True  → Matching will be case-sensitive ("Python" != "python").
                             #   False → Matching will ignore letter case ("Python" == "python").
                             # Purpose: Makes it easy to toggle mode without changing code elsewhere.

SESSION_PATH = "session.json"   #for saving session

MAX_HISTORY = 50  # keep last 50 searches only