import re                         # for regex searching and substitution


def whole_word_pattern(keyword):
    """
    Build a regex pattern that matches 'keyword' as a whole word.
    Uses re.escape to make the keyword safe if it contains regex-special characters.
    Why: Avoids 'main' matching 'remained' etc.
    """
    return r'\b' + re.escape(keyword) + r'\b'





