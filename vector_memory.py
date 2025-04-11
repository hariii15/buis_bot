import os
import json
import supabase
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
import google.generativeai as genai
from supabase import create_client, Client

class VectorMemory:
    def __init__(self):
        load_dotenv()

        # Initialize Supabase
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

        # Initialize Gemini Embeddings
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.embedding_model = genai.embed_content

    def embed_text(self, text: str) -> list:
        try:
            response = self.embedding_model(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error embedding text: {e}")
            return [0.0] * 768

    def store_user_context(self, user_id: str, context: dict):
        context_str = json.dumps(context)
        embedding = self.embed_text(context_str)
        # Store in Supabase
        self.supabase.table("user_contexts").upsert({
            "user_id": user_id,
            "embedding": embedding,
            "context": context_str
        }).execute()

    def get_user_context(self, user_id: str) -> dict:
        result = self.supabase.table("user_contexts").select("context").eq("user_id", user_id).execute()
        if result.data:
            return json.loads(result.data[0]["context"])
        return {}
