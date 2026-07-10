import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def generate_answer(query, context_chunks, memory=None, book_title: str = "the book"):
    """
    Generates an answer using Groq (Llama 3.3 70B).
    book_title: dynamically injected so the AI knows what book it's helping with.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ Error: Missing GROQ_API_KEY in .env file."

    client = Groq(api_key=api_key)

    # 1. Context from RAG
    context_text = "\n\n".join(context_chunks) if context_chunks else "No relevant excerpts found."

    # 2. Conversation memory
    memory_text = ""
    if memory:
        recent = memory.get_context(limit=6)
        if recent:
            memory_text = "\n\n--- RECENT CONVERSATION ---\n"
            for msg in recent:
                memory_text += f"{msg['role'].upper()}: {msg['content']}\n"

    # 3. Dynamic system prompt — uses the real book title
    system_prompt = (
        f"You are BookFriend, a helpful AI assistant for the novel '{book_title}'.\n"
        "Answer the user's question strictly based on the provided context excerpts below.\n"
        "If the answer isn't in the context, say you don't know. Do not make things up.\n"
        "Keep answers concise, clear, and spoiler-safe based on the context given.\n"
    )

    user_content = (
        f"{memory_text}\n"
        f"--- CONTEXT EXCERPTS ---\n{context_text}\n"
        "------------------------\n\n"
        f"USER QUESTION: {query}"
    )

    # 4. Generate
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.5,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq Error: {str(e)}"