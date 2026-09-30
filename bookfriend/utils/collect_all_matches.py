""" What this function is (one brutal simple line)
collect_all_matches(...) =
Scan all .txt chapters in a folder → split into sentences →
find exact + fuzzy keyword hits → return a list of Match records with snippet + metadata.
This is your search indexer.
"""

import os  # for file and path handling, launching OS commands
import re  # for regex searching and substitution
from collections import namedtuple  # for simple structured storage (Match records)

from rapidfuzz import fuzz

from .make_snippet import make_snippet
from .whole_word_pattern import whole_word_pattern


Match = namedtuple("Match", [
    "file",     # filename where match occurred (e.g., 'chapter0001.txt')
    "sentence", # full sentence text containing the match (original unmodified)
    "start",    # start character index of the match within the sentence (int)
    "end",      # end character index of the match within the sentence (int)
    "keyword",  # the keyword string that matched (from user input)
    "snippet",  # short snippet string built around the match for preview
    "is_fuzzy"  # NEW: True if this match came from fuzzy search
])


def collect_all_matches(folder_path, keywords, case_sensitive=False, fuzzy=False, threshold=80, chapter_filter = None, valid_range = None):
    #print(f"[DEBUG] collect_all_matches called with fuzzy={fuzzy}")

    """
    Walk through all '.txt' files in folder_path, split each file into sentences,
    find all whole-word matches for ANY keyword, and collect Match records.
    Returns a list of Match objects.
    Why:
      - We collect everything first so we can show a preview list, counts, and support navigation.
      - Pre-compiling regex patterns speeds up repeated matching across many sentences.
    """
    # Ensure the folder exists before proceeding — helpful early error for the user.
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    matches = []  # list to collect Match records

    # Pre-compile regex patterns per keyword (faster when scanning many sentences)
    patterns = {}
    for kw in keywords:
        # compile pattern with correct flags (case-insensitive unless case_sensitive True)
        patterns[kw] = re.compile(whole_word_pattern(kw), 0 if case_sensitive else re.IGNORECASE)

    # Walk files in sorted order for stable, predictable output across runs
    for fname in sorted(os.listdir(folder_path)):
        # Only consider files ending with .txt (case-insensitive)
        if not fname.endswith(".txt"):
            continue
        # ✅ Filter by chapter_range if set
        if valid_range:
            # Extract number from e.g. "chapter0005.txt"
            try:
                num = int(''.join(ch for ch in fname if ch.isdigit()))
                if num not in valid_range:
                    continue  # skip this file
            except ValueError:
                continue  # if filename has no digits, skip
    #  Skip if chapter filter is set and this file doesn't match
        if chapter_filter:
            if chapter_filter not in fname.lower() and chapter_filter.zfill(8) not in fname.lower():
                continue

        file_path = os.path.join(folder_path, fname)
        # Read file content using UTF-8 encoding, skipping files that fail to open
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except Exception as e:
            print(f"[ERROR] Could not read '{file_path}': {e}")
            continue

        # Split into sentences by a simple rule: punctuation (.!? ) followed by whitespace
        # This is the same rule used earlier in your project and is good enough for quick previews.
        sentences = re.split(r'(?<=[.!?])\s+', content)

        # For each sentence we check each compiled pattern using finditer to capture positions
        for sentence in sentences:
            sentence_lower = sentence.lower()  # helpful for fuzzy matching

            for kw, pat in patterns.items():
                #print(f"[DEBUG] Regex for '{kw}': {pat.pattern}")

                # --- Exact regex matches ---
                for m in pat.finditer(sentence):
                    start, end = m.start(), m.end()
                    snippet = make_snippet(sentence, start, end)
                    matches.append(Match(
                        file=fname,
                        sentence=sentence,
                        start=start,
                        end=end,
                        keyword=kw,
                        snippet=snippet,
                        is_fuzzy=False  # <-- exact
                    ))

                # --- Fuzzy matches ---
                # --- Fuzzy matches (word-level) ---
                if fuzzy:
                    words = re.findall(r"\w+", sentence_lower)  # split into words, keep only alphanum
                    for word in words:
                        score = fuzz.ratio(kw.lower(), word)
                        if score >= threshold and not pat.search(sentence):
                            # Find word position in original sentence for snippet
                            start = sentence_lower.find(word)
                            end = start + len(word)
                            snippet = make_snippet(sentence, start, end)
                            matches.append(Match(
                                file=fname,
                                sentence=sentence,
                                start=start,
                                end=end,
                                keyword=f"{kw} (fuzzy {score}%)",
                                snippet=snippet,
                                is_fuzzy=True
                            ))

    # Return all matched records (could be empty list if no matches)
    return matches