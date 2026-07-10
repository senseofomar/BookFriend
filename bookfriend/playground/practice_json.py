import json
from pathlib import Path

#1. some fake data
session_data = {
    "last_query": "Klein",
    "matches": ["Klein Moretti", "Klein muttered..."],
    "current_index": 1
}
#2. Save it
with open('test.json', 'w', encoding='utf-8')as f:
    json.dump(session_data, f, indent = 2)

#3. Load it
with open('test.json', 'r', encoding='utf-8') as f:
    loaded = json.load(f)

#4. Print it
print(loaded)


def save_session(data:dict, path:str)->None:
    with open(path, 'w',encoding='utf-8')as f:
        json.dump(data, f, indent= 2)

def load_session(path:str)->dict:
    if not Path(path).exists():
        return {}
    with open(path, 'r',encoding='utf-8')as f:
        return json.load(f)

def reset_session(path:str)->None:
    Path(path).unlink(missing_ok =True)



