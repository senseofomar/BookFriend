import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(query: str, context_chunks: list, memory, book_title: str) -> str:
    context_str = "\n\n".join(context_chunks)
    history = memory.get_context()

    system_prompt = (
        f"You are BookFriend, an expert AI reading assistant for the book '{book_title}'.\n\n"
        "CORE RULES:\n"
        "1. STRICT GROUNDING: Answer ONLY using the 'Excerpt Context' provided below. If the answer is not in the context, say 'I'm sorry, I don't have enough information from the current chapters to answer that.'\n"
        "2. NO SPOILERS: Never mention plot points, characters, or events from future chapters. Only discuss what has been revealed in the provided excerpts.\n"
        "3. CITATIONS: When possible, reference which part of the context you are using (e.g., 'According to the excerpt...').\n"
        "4. NO OUTSIDE KNOWLEDGE: Do not use your internal knowledge about the book. Only use the provided context.\n"
        "5. PERSONALITY: Be helpful, encouraging, and scholarly. Maintain the tone of a reading companion."
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add chat history (limited to last 6 to save tokens/context)
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current query with context
    messages.append({
        "role": "user",
        "content": f"--- START OF EXCERPT CONTEXT ---\n{context_str}\n--- END OF EXCERPT CONTEXT ---\n\nQuestion: {query}"
    })

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2, # Lower temperature for more deterministic/grounded answers
    )

    return completion.choices[0].message.content


def detect_intent(query: str) -> str:
    """Classifies the user query intent to decide between RAG or Global Summary."""
    prompt = (
        "Classify the following user query into one of two categories: 'SUMMARY' or 'QUERY'.\n"
        "- SUMMARY: The user wants an overview, a summary of the whole book, or a recap of everything so far.\n"
        "- QUERY: The user is asking a specific question about a character, event, or detail.\n\n"
        f"Query: \"{query}\"\n\n"
        "Respond with ONLY the category name."
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3-8b-8192", # Use a smaller, faster model for classification
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        intent = completion.choices[0].message.content.strip().upper()
        return "SUMMARY" if "SUMMARY" in intent else "QUERY"
    except Exception:
        return "QUERY" # Default to QUERY on failure
