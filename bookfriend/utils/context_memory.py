
from difflib import get_close_matches

def recall_last_search(session_data):
    if not session_data["search_history"]:
        return None
    return session_data["search_history"][-1]

def suggest_related(session_data, current_keywords):
    all_keywords = [kw for hist in session_data["search_history"] for kw in hist[0]]
    related = []
    for kw in current_keywords:
        related += get_close_matches(kw, all_keywords, n=2, cutoff=0.6)
    return list(set(related))
