# Interactive multi-keyword search with:
#  - distinct colors per keyword
#  - context preview (snippet around the match)
#  - list of matches (file + snippet)
#  - interactive navigation: n (next), p (prev), number (jump), o (open in editor), q (quit)

import os  # for file and path handling, launching OS commands

from .highlight import highlight_sentence_with_colors, CHAPTERS_FOLDER
from .open_in_pycharm import open_in_pycharm, compute_match_file_line


def interactive_navigation(matches, keywords, kw_color_map, case_sensitive=False):
    """
    Presents matches previews and allows interactive navigation.
    Commands:
      - n or Enter: next match
      - p          : previous match
      - number     : jump to that match number (1-based)
      - o          : open current match file in PyCharm (or fallback)
      - q          : quit navigation
      - f          : filter results by word (case-insensitive)
    The function prints the highlighted sentence and a snippet for each match.
    """
    # If no matches, inform and return early
    if not matches:
        print("No matches found.")
        return

    # Print a summary list of matches (index + filename + colored snippet)
    print(f"\nFound {len(matches)} matches. Showing previews (context snippets):\n")
    for i, m in enumerate(matches, start=1):
        # Use color highlighting in the preview snippet
        preview = highlight_sentence_with_colors(m.snippet, [m], keywords, kw_color_map, case_sensitive=case_sensitive)
        print(f"{i:04d}. {m.file} - {preview}")


    # Navigation index (0-based)
    idx = 0
    print("\nNavigation commands: 'n' = next, 'p' = previous, a number = jump, 'o' = open in editor, 'q' = quit")

    # Loop until user quits
    while True:
        current = matches[idx]
        # Show full sentence with colors applied
        # When showing full sentence
        full_colored = highlight_sentence_with_colors(current.sentence, [current], keywords, kw_color_map,
                                                      case_sensitive=case_sensitive)
        print("\n" + "="*80)
        print(f"Match {idx+1}/{len(matches)}  —  File: {current.file}  —  Keyword: {current.keyword}")
        print("- Full sentence (highlighted):")
        # print full colored sentence; strip to remove leading/trailing blank lines
        print(full_colored.strip())
        print("- Snippet:")
        print(current.snippet)
        print("="*80)

        # Read command from user
        cmd = input("\nNavigation [n=next, p=prev, o=open, f=filter, q=quit]: ").strip()

        # Quit commands
        if cmd.lower() in ("q", "quit", "exit"):
            print("Exiting navigation.")
            break

        # Next or empty (Enter) -> next match (circular)
        if cmd.lower() in (" ", "n"):
            idx = (idx + 1) % len(matches)
            continue

        # Previous -> previous match (circular)
        if cmd.lower() == "p":
            idx = (idx - 1) % len(matches)
            continue

        if cmd == "f":
            sub = input("Filter word: ").strip().lower()
            filtered = [m for m in matches if sub in m.sentence.lower()]
            if filtered:
                matches = filtered
                idx = 0
            else:
                print("⚠️ No results after filter.")
            continue

        # Open in editor: compute line and open
        if cmd.lower() == "o":
            # Compute file path relative to CHAPTERS_FOLDER
            file_path = os.path.join(CHAPTERS_FOLDER, current.file)
            # Compute line number in file for match start
            line, col = compute_match_file_line(file_path, current.sentence, current.start)
            if line:
                print(f"Opening {current.file} at line {line} (column {col}) in editor...")
            else:
                print(f"Opening {current.file} in editor (line not found precisely; opening file).")

            ok = open_in_pycharm(file_path, line=line, column=col)
            if not ok:
                print("Failed to open in PyCharm/OS default editor. Check your 'charm' launcher or system opener.")
            # Keep current match displayed (do not change idx)
            continue

        # Try to interpret as a number to jump to
        try:
            num = int(cmd)
            if 1 <= num <= len(matches):
                idx = num - 1
            else:
                print(f"Number out of range. Enter 1..{len(matches)}")
        except ValueError:
            # Unrecognized command
            print("Unknown command. Use 'n', 'p', a number, 'o' to open file, or 'q' to quit.")