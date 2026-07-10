

import os                         # for file and path handling, launching OS commands
import sys                        # for sys.exit and platform detection
import subprocess                 # for launching external programs (like editor)
from shutil import which          # to check if an executable exists on PATH


# If you prefer a different command to open PyCharm, change this variable.
# Typical name for the PyCharm command-line launcher is 'charm'.
# You can create it in PyCharm: Tools → Create Command-line Launcher... → name 'charm'
PYCHARM_LAUNCHER_CMD = "charm"


def open_in_pycharm(file_path, line=None, column=None):
    """
    Try to open the given file in PyCharm at the given line (and optional column).
    Strategy:
      1) If 'charm' (PyCharm command-line launcher) is available on PATH, use it.
         Example: charm --line 120 /path/to/file (some versions accept file:line)
      2) If not available, attempt general OS open:
         - On Windows: os.startfile(file_path)
         - On macOS: 'open' command via subprocess
         - On Linux: 'xdg-open' via subprocess
    Notes:
      - PyCharm's CLI 'charm' behavior may vary by version/OS. Users are encouraged to set up 'charm'.
      - If 'charm' isn't present, we still open the file in the system default editor as a fallback.
    Returns True if a command was launched successfully, False otherwise.
    """
    # Normalize path
    file_path = os.path.abspath(file_path)

    # 1) If `charm` exists on PATH, try to use it with line number if supported.
    if which(PYCHARM_LAUNCHER_CMD):
        try:
            # Some 'charm' versions accept 'charm file:line' or 'charm --line line file'
            # We'll try both forms; if first fails, second may work.
            if line:
                # Try "charm file:line" first (common)
                try:
                    subprocess.Popen([PYCHARM_LAUNCHER_CMD, f"{file_path}:{line}"])
                    return True
                except Exception:
                    # Fall back to explicit --line if available
                    try:
                        subprocess.Popen([PYCHARM_LAUNCHER_CMD, "--line", str(line), file_path])
                        return True
                    except Exception:
                        # Give up on charm method
                        pass
            else:
                # No line number requested, just open file
                subprocess.Popen([PYCHARM_LAUNCHER_CMD, file_path])
                return True
        except Exception:
            # If anything fails with charm, we will fall back below
            pass

    # 2) Platform-specific fallback: open file with system default app
    try:
        if sys.platform.startswith("win"):
            # Windows: os.startfile will use the registered app for the file type
            os.startfile(file_path)
            return True
        elif sys.platform == "darwin":
            # macOS: 'open' command
            subprocess.Popen(["open", file_path])
            return True
        else:
            # Assume Linux or similar: use xdg-open
            subprocess.Popen(["xdg-open", file_path])
            return True
    except Exception as e:
        print(f"[ERROR] Could not open file with system opener: {e}")
        return False



def compute_match_file_line(file_path, sentence, match_start_in_sentence):
    """
    Compute the 1-based line number in the file corresponding to the start of the match.
    Approach:
      - Read full file text.
      - Find the first occurrence of the sentence in the file (we assume the sentence text is unique enough).
      - Compute the absolute character offset where the match occurs:
          file_offset = sentence_offset_in_file + match_start_in_sentence
      - Convert file_offset to line number by counting newlines before that offset.
    Returns (line_number, column) both 1-based, or (None, None) if we can't find the sentence.
    Why:
      - Editors usually accept a line number to jump to; we compute it precisely.
      - Using the sentence search is robust if sentence boundaries are preserved in the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            full = fh.read()
    except Exception as e:
        # If we can't open the file, return None values (caller will handle fallback)
        print(f"[ERROR] Could not read file to compute line number: {e}")
        return None, None

    # Try to locate the sentence inside the full file text
    # We attempt an exact match; if the sentence is present multiple times, this picks the first.
    idx = full.find(sentence)
    if idx == -1:
        # As a fallback, try a best-effort: search by the snippet (shorter string)
        # This helps if sentence splitting trimmed whitespace differently.
        snippet = sentence.strip()
        idx = full.find(snippet)
        if idx == -1:
            return None, None

    # file_offset is where the sentence starts in the file.
    file_offset = idx + match_start_in_sentence

    # Compute 1-based line number by counting '\n' up to file_offset
    line_number = full.count("\n", 0, file_offset) + 1
    # Compute column by finding the last newline before file_offset
    last_nl = full.rfind("\n", 0, file_offset)
    if last_nl == -1:
        column = file_offset + 1  # no newline before → column is offset+1 (1-based)
    else:
        column = file_offset - last_nl
    return line_number, column