# utils/command_router.py

from .config import SESSION_PATH
from . import session_utils
from bookfriend.utils.context_memory import recall_last_search
from bookfriend.utils.memory_tools import recall_recent_queries, summarize_memory

def handle_command(raw_input_val, session_data, chapter_range, semantic_index, semantic_mapping, memory):
    """Handle user commands from main() and return (handled, updated_chapter_range)."""

    cmd = raw_input_val.lower()

    # Exit or quit
    if cmd in ("q", "quit", "exit"):
        return "exit", chapter_range

    # Search history
    if cmd == "search-history":
        if not session_data["search_history"]:
            print("No searches yet.")
        else:
            print("Last searches:")
            for i, (keys, chap, fuzzy) in enumerate(session_data["search_history"], 1):
                chap_label = chap if chap else "all"
                fuzzy_label = "fuzzy" if fuzzy else "exact"
                print(f"{i}. {', '.join(keys)} [{chap_label}, {fuzzy_label}]")
        return True, chapter_range

    # Save history
    if cmd == "save-history-now":
        session_utils.save_session(session_data, SESSION_PATH)
        print("✅ Session saved.")
        return True, chapter_range

    # Clear history
    if cmd == "clear-history":
        confirm = input("⚠️ Are you sure? (y/n): ")
        if confirm.lower() == "y":
            session_data["search_history"].clear()
            print("🧹 History cleared.")
        return True, chapter_range

    # Clear Context
    if cmd == "forget":
        memory.clear()
        print("🧹 Memory cleared. Starting a new conversation.")
        return True, None

    # Stats
    if cmd == "stats":
        print(f"📊 Total searches ever: {session_data['total_search_count']}")
        print(f"📊 Saved history size: {len(session_data['search_history'])}")
        return True, chapter_range

    # Favorites
    if cmd == "fav-add":
        if not session_data["search_history"]:
            print("⚠️ No search yet to add to favorites.")
        else:
            last_search = session_data["search_history"][-1]
            if last_search not in session_data["favorites"]:
                session_data["favorites"].append(last_search)
                print(f"✅ Added to favorites: {last_search}")
            else:
                print("ℹ️ Already in favorites.")
        return True, chapter_range

    if cmd == "fav-list":
        if not session_data["favorites"]:
            print("⭐ No favorites yet.")
        else:
            print("\n⭐ Favorites:")
            for i, fav in enumerate(session_data["favorites"], 1):
                keys, chap, fuzzy = fav
                chap_label = chap if chap else "all"
                fuzzy_label = "fuzzy" if fuzzy else "exact"
                print(f"{i}. {', '.join(keys)} [{chap_label}, {fuzzy_label}]")
        return True, chapter_range

    # Range
    if cmd == "set-range":
        try:
            raw = input("Enter start and end chapter (e.g., 1 50): ").strip()
            start, end = map(int, raw.split())
            chapter_range = [start, end]
            session_data["chapter_range"] = chapter_range
            session_utils.save_session(session_data, SESSION_PATH)
            print(f"✅ Range set: {start} → {end}")
        except ValueError:
            print("⚠️ Invalid input format. Please enter two numbers separated by a space (e.g., '1 50').")
        except Exception as e:
            print(f"⚠️ Error setting range: {e}")
        return True, chapter_range

    if cmd == "show-range":
        if chapter_range:
            print(f"Current range: {chapter_range[0]} → {chapter_range[1]}")
        else:
            print("No range set.")
        return True, chapter_range

    if cmd == "clear-range":
        chapter_range = None
        session_data["chapter_range"] = None
        session_utils.save_session(session_data, SESSION_PATH)
        print("🗑️ Range cleared. Searching all chapters.")
        return True, chapter_range

    # Recall last
    if cmd == "recall-last":
        last = recall_last_search(session_data)
        if not last:
            print("⚠️ No previous search found.")
        else:
            keys, chap, fuzzy = last
            print(f"Last search: {keys} [{chap or 'all'}, {'fuzzy' if fuzzy else 'exact'}]")
        return True, chapter_range

    # 🧠 Memory commands
    if cmd == "recall-memory":
        recent = recall_recent_queries(memory.get_all())
        if not recent:
            print("🧠 No recent queries found.")
        else:
            print("\n🧠 Recent Queries:")
            for i, q in enumerate(recent, 1):
                print(f"{i}. {q}")
        return True, chapter_range

    if cmd == "summarize-memory":
        summary = summarize_memory(memory.get_all())
        print("\n🧩 Memory Summary:\n")
        print(summary)
        return True, chapter_range

    # === REMOVED: Semantic Search Block ===
    # It is now handled exclusively by main.py to enable the Spoiler Shield.

