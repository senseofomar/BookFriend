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
from bookfriend.utils.command_router import handle_command
from bookfriend.utils.context_memory import suggest_related
from bookfriend.memory import ChatMemory
from bookfriend.utils.collect_all_matches import collect_all_matches
from bookfriend.utils.config import CASE_SENSITIVE_MODE, SESSION_PATH, MAX_HISTORY
from bookfriend.utils.export_to_csv import export_to_csv
from bookfriend.utils.highlight import build_keyword_color_map, CHAPTERS_FOLDER
from bookfriend.utils.interactive_navigation import interactive_navigation
from bookfriend.utils import session_utils
from bookfriend.utils.semantic_utils import semantic_search
from bookfriend.utils.answer_generator import generate_answer, detect_intent
from bookfriend.services.summarizer import generate_global_summary

def main():
    """Main controller for bookfriend CLI."""
    # === Load or Initialize User Session ===
    session_data = session_utils.load_session(SESSION_PATH)
    session_data.setdefault("search_history", [])
    session_data.setdefault("total_search_count", 0)
    session_data.setdefault("favorites", [])
    session_data.setdefault("current_book_id", None)

    # Load chapter range from session if it exists
    chapter_range = session_data.get("chapter_range", [1, 10])

    # === Display Mode Info ===
    mode_label = "CASE-SENSITIVE" if CASE_SENSITIVE_MODE else "CASE-INSENSITIVE"
    print(f"\n📘 bookfriend — Multi-keyword & Semantic Search ({mode_label} mode)")
    print("💡 Type 'q' or 'quit' to exit, 'forget' to clear memory.")
    print("💡 Use 'set-book <id>' to select a book from Supabase.\n")

    if not session_data["current_book_id"]:
        print("⚠️ No book selected. Semantic search will be disabled until you run 'set-book <id>'.")

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
        if raw_input_val.startswith("set-book "):
            bid = raw_input_val.split(" ", 1)[1].strip()
            session_data["current_book_id"] = bid
            print(f"✅ Current book set to: {bid}")
            continue

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
            book_id = session_data["current_book_id"]
            if not book_id:
                print("❌ No book selected. Use 'set-book <id>' first.")
                continue

            query = raw_input_val.split("semantic:", 1)[1].strip()

            # 1. Intent Detection
            intent = detect_intent(query)
            user_max_chapter = chapter_range[1] if chapter_range else 999999

            if intent == "SUMMARY":
                print(f"🤖 Generating global summary up to Ch. {user_max_chapter}...")
                answer = generate_global_summary(book_id, "Selected Book", user_max_chapter)
                print(f"\n📚 Summary:\n{answer}\n")
                memory.add("assistant", answer)
                continue

            # 2. Semantic Search
            print(f"🤔 Searching Supabase for: '{query}' (Limit: Ch. {user_max_chapter})...")
            final_results = semantic_search(
                query=query,
                book_id=book_id,
                chapter_limit=user_max_chapter,
                top_k=5
            )

            if not final_results:
                print(f"🔒 Spoiler Shield Active or No Matches found up to Chapter {user_max_chapter}.")
                continue

            print(f"\n🔎 Top Semantic Matches:\n")
            for fname, chunk, dist in final_results:
                print(f"[{fname}] (score={dist:.2f}) → {chunk[:150]}...\n")

            # --- Generate Answer ---
            top_chunks = [chunk for _, chunk, _ in final_results]
            print("\n🤖 Thinking...\n")

            try:
                answer = generate_answer(query, top_chunks, memory=memory, book_title="the book")
                memory.add("assistant", answer)
                print(answer)
                print("\n✅ Done.\n")
            except Exception as e:
                print(f"⚠️ Answer generation failed: {e}")

            continue

        # ======================================================
        # === KEYWORD SEARCH MODE (Local) ===
        # ======================================================
        keywords = [k.strip() for k in raw_input_val.split(",") if k.strip()]
        if not keywords:
            continue

        print("📂 Local Keyword Search (Requires 'chapters/' folder)")
        # Restrict to Chapter Range
        valid_range = range(chapter_range[0], chapter_range[1] + 1) if chapter_range else None

        matches = collect_all_matches(
            CHAPTERS_FOLDER,
            keywords,
            case_sensitive=CASE_SENSITIVE_MODE,
            fuzzy=False,
            chapter_filter=None,
            valid_range=valid_range
        )

        if not matches:
            print("⚠️ No matches found in local chapters.")
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
