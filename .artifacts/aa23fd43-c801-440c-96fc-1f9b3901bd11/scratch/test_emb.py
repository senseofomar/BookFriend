import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "gemini-embedding-001"

def test():
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=["Hello world"],
    )
    print(f"Dimensions: {len(response.embeddings[0].values)}")

if __name__ == "__main__":
    test()
