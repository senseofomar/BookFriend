import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(query: str, context_chunks: list, memory, book_title: str) -> str:
    context_str = "\n\n".join(context_chunks)
    history = memory.get_context()

    messages = [
        {
            "role": "system",
            "content": f"You are BookFriend, an assistant helping a reader discuss '{book_title}'. "
                       f"Only use the provided excerpt context up to their reading chapter. "
                       f"Do not spoil upcoming chapters."
        }
    ]

    # Add chat history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current query with context
    messages.append({
        "role": "user",
        "content": f"Excerpt Context:\n{context_str}\n\nQuestion: {query}"
    })

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
    )

    return completion.choices[0].message.content