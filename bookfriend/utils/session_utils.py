import json
from pathlib import Path


def save_session(data:dict, path:str)->None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent = 2)


def load_session(path:str)->dict:

    if not Path(path).exists():
        return{}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:  # file exists but is empty
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        print(f"[WARN] Session file corrupted → starting fresh.")
        return {}

def reset_session(path:str)->None:
    Path(path).unlink(missing_ok=True)