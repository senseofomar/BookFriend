from bookfriend.utils.highlight import CONTEXT_CHARS


def make_snippet(sentence, start, end, context_chars=CONTEXT_CHARS):

    s = max(0, start - context_chars)
    e = min(len(sentence), end + context_chars)
    prefix = "..." if s > 0 else ""
    suffix = "..." if e < len(sentence) else ""
    snippet = prefix + sentence[s:e].strip() + suffix
    return snippet