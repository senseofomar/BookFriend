import json
import os
from difflib import get_close_matches

FAQ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faq_data.json")

def get_faq_answer(query: str) -> str:
    """Simple keyword/fuzzy matcher for the FAQ data."""
    if not os.path.exists(FAQ_PATH):
        return None

    try:
        with open(FAQ_PATH, "r") as f:
            faqs = json.load(f)

        questions = [f["question"] for f in faqs]
        matches = get_close_matches(query, questions, n=1, cutoff=0.6)

        if matches:
            for f in faqs:
                if f["question"] == matches[0]:
                    return f["answer"]

        # Fallback: check if any keyword matches
        query_words = set(query.lower().split())
        for f in faqs:
            q_words = set(f["question"].lower().split())
            if len(query_words.intersection(q_words)) >= 2:
                return f["answer"]

        return None
    except Exception:
        return None
