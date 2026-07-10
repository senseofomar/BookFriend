"""
bookfriend — Intelligent Semantic + Keyword Search CLI
------------------------------------------------------
This module orchestrates the main CLI flow for bookfriend.
"""

from dotenv import load_dotenv
import os
import sys

# === Environment Setup ===
load_dotenv()

# === Internal Imports ===
# (Adjusted to work locally from the bookfriend folder)
from utils.command_router import handle_command
from utils.context_memory import suggest_related
from memory import ChatMemory
from utils.collect_all_matches import collect_all_matches
from utils.config import CASE_SENSITIVE_MODE, SESSION_PATH, MAX_HISTORY
from utils.export_to_csv import export_to_csv
from utils.highlight import build_keyword_color_map, CHAPTERS_FOLDER
from utils.interactive_navigation import interactive_navigation
from utils import session_utils
from utils.semantic_utils import semantic_search
from utils.answer_generator import generate_answer
from bookfriend.api import app

def main():
    """Main controller for bookfriend CLI."""
    # === Load or Initialize User Session ===
    session_data = session_utils.load_session(SESSION_PATH)
    session_data.setdefault("search_history", [])
    session_data.setdefault("total_search_count", 0)
    session_data.setdefault("favorites", [])

    # Load chapter range from session if it exists
    chapter_range = session_data.get("chapter_range", None)
    search_this_session = 0

    # === Verify Chapter Data Directory ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "chapters")
    if not os.path.isdir(folder):
        print(f"[❌ ERROR] Missing folder: {folder}")
        print("👉 Run 'ingest.py' first!")
        sys.exit(1)

    # === Display Mode Info ===
    mode_label = "CASE-SENSITIVE" if CASE_SENSITIVE_MODE else "CASE-INSENSITIVE"
    print(f"\n📘 bookfriend — Multi-keyword & Semantic Search ({mode_label} mode)")
    print("💡 Type 'q' or 'quit' to exit, 'forget' to clear memory.\n")

    if session_data["search_history"]:
        print(f"📂 Previous session loaded. {len(session_data['search_history'])} past searches available.\n")

    if chapter_range:
        print(f"🛡️ Spoiler Shield ACTIVE: Limited to Chapters {chapter_range[0]} - {chapter_range[1]}\n")


    # === Initialize Conversation Memory ===
    memory = ChatMemory(max_messages=10)

    # === CLI Main Loop ===
    while True:
        try:
            raw_input_val = input("\n🔍 Enter keyword(s) or command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🛑 Exiting.")
            session_utils.save_session(session_data, SESSION_PATH)
            break

        if not raw_input_val:
            continue

        # --- Handle Commands ---
        handled, chapter_range = handle_command(
            raw_input_val,
            session_data,
            chapter_range,
            memory
        )

        if handled == "exit":
            break
        elif handled:
            continue

        # --- Update Memory with User Query ---
        memory.add("user", raw_input_val)

        # ======================================================
        # === SEMANTIC SEARCH MODE (With Spoiler Shield) ===
        # ======================================================
        if raw_input_val.startswith("semantic:"):
            if not semantic_index:
                print("❌ Semantic index is missing. Cannot perform AI search.")
                continue

            query = raw_input_val.split("semantic:", 1)[1].strip()

            # 1. Retrieve MORE results than needed (to allow filtering)
            raw_results = semantic_search(query, semantic_index, semantic_mapping, top_k=50)

            if not raw_results:
                print("⚠️ No semantic results found.")
                continue

            # 2. Apply Spoiler Shield
            user_max_chapter = chapter_range[1] if chapter_range else 999999
            safe_results = []

            for fname, chunk, dist in raw_results:
                # Extract number from filename (e.g. "chapter_005.txt" -> 5)
                try:
                    chap_num = int(''.join(filter(str.isdigit, fname)))
                    if chap_num <= user_max_chapter:
                        safe_results.append((fname, chunk, dist))
                except ValueError:
                    # If no number found, keep it safe
                    safe_results.append((fname, chunk, dist))

            # 3. Take Top 5 SAFE results
            final_results = safe_results[:5]

            if not final_results:
                print(f"🔒 Spoiler Shield Active! Found matches, but ALL were beyond Chapter {user_max_chapter}.")
                print("👉 Try increasing your range with 'set-range'.")
                continue

            print(f"\n🔎 Top Semantic Matches (Filtered to Ch. {user_max_chapter}):\n")
            for fname, chunk, dist in final_results:
                print(f"[{fname}] (score={dist:.2f}) → {chunk[:200]}...\n")

            # --- Generate Answer ---
            top_chunks = [chunk for _, chunk, _ in final_results[:3]]
            print("\n🤖 bookfriend’s Interpretation:\n")
            print("🤔 Thinking...\n")

            try:
                answer = generate_answer(query, top_chunks, memory=memory)
                memory.add("assistant", answer)
                print(answer)
                print("\n✅ Done.\n")
            except Exception as e:
                print(f"⚠️ Answer generation failed: {e}")

            continue

        # ======================================================
        # === KEYWORD SEARCH MODE ===
        # ======================================================
        keywords = [k.strip() for k in raw_input_val.split(",") if k.strip()]
        if not keywords:
            print("⚠️ Please enter at least one keyword.")
            continue

        mode = input("📂 Search in (a)ll chapters or (s)pecific? [a/s]: ").strip().lower()
        chapter_filter = None
        if mode == "s":
            chapter_filter = input("Enter part of chapter filename: ").strip() or None

        related = suggest_related(session_data, keywords)
        if related:
            print(f"💡 Related Past Keywords: {', '.join(related)}")

        use_fuzzy = input("Enable fuzzy search? (y/n): ").strip().lower() in ("y", "yes")

        session_data["total_search_count"] += 1
        session_data["search_history"].append((keywords, chapter_filter, use_fuzzy))
        if len(session_data["search_history"]) > MAX_HISTORY:
            session_data["search_history"].pop(0)

        # Restrict to Chapter Range
        valid_range = range(chapter_range[0], chapter_range[1] + 1) if chapter_range else None

        matches = collect_all_matches(
            CHAPTERS_FOLDER,
            keywords,
            case_sensitive=CASE_SENSITIVE_MODE,
            fuzzy=use_fuzzy,
            chapter_filter=chapter_filter,
            valid_range=valid_range
        )

        if not matches:
            print("⚠️ No matches found.")
            continue

        kw_color_map = build_keyword_color_map(keywords)
        interactive_navigation(matches, keywords, kw_color_map)
        export_to_csv(matches, "recent_search_results.csv")

        session_utils.save_session(session_data, SESSION_PATH)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 Fatal Error: {e}")
        sys.exit(1)