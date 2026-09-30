import re

from bookfriend.utils.config import CASE_SENSITIVE_MODE


def keyword_in_sentence(keywords, sentence):
    """
    Checks if ANY keyword exists in the given sentence as a WHOLE WORD.
    Returns True if at least one keyword is found, otherwise False.
    Purpose: Avoids partial matches and ensures quick yes/no decision before doing more work.
    """
    # Same case-sensitivity setup as before.
    flags = 0 if CASE_SENSITIVE_MODE else re.IGNORECASE

    # Loop through all user keywords.
    for keyword in keywords:
        # Same regex whole-word boundary technique to avoid false positives inside other words.
        pattern = r'\b' + re.escape(keyword) + r'\b'

        # re.search → Searches for the first occurrence of the pattern in the sentence.
        # If found → return True immediately (no need to check more keywords).
        if re.search(pattern, sentence, flags):
            return True

    # If loop finishes without finding a match → return False.
    return False
