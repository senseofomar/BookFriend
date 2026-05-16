from collections import deque

class ChatMemory:
    """
    Handles short-term conversation memory for bookfriend.
    - Stores user and assistant messages in chronological order
    - Provides limited context for continuity
    - Can reset or export full history if needed
    """

    def __init__(self, max_messages=20):
        # Use deque for efficient pops when max limit exceeded
        self.messages = deque(maxlen=max_messages)

    def add(self, role: str, content: str):
        """
        Add a new message to memory.
        role: 'user' | 'assistant'
        """
        if role not in ("user", "assistant"):
            raise ValueError("Role must be 'user' or 'assistant'")
        self.messages.append({"role": role, "content": content})

    def get_context(self, limit: int = 6):
        """
        Return the most recent 'limit' messages as a list
        (used for continuity in LLM calls).
        """
        return list(self.messages)[-limit:]

    def clear(self):
        """Completely clear the conversation history."""
        self.messages.clear()

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        return f"<ChatMemory size={len(self.messages)}>"
