from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient  # Import Hugging Face InferenceClient

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://buiss-bot.vercel.app"}})  # Allow requests from localhost:5173

# Retrieve Supabase credentials and Hugging Face API key from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not HUGGINGFACE_API_KEY:
    raise ValueError("SUPABASE_URL, SUPABASE_KEY, and HUGGINGFACE_API_KEY must be set in the .env file")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Hugging Face InferenceClient
client = InferenceClient(provider="novita", api_key=HUGGINGFACE_API_KEY)

def get_user_contexts(user_id: str):
    """
    Retrieve all context entries for a user from the unified user_contexts table.
    """
    try:
        response = supabase.table("user_contexts").select("*").eq("user_id", user_id).order("created_at").execute()
        if response.data:
            return response.data
        return []
    except Exception as e:
        print(f"Error retrieving user contexts: {e}")
        return []

def store_user_context(user_id: str, prompt: str, response: str, embedding: dict = None):
    """
    Store a new context entry in the unified user_contexts table.
    """
    try:
        supabase.table("user_contexts").insert({
            "user_id": user_id,
            "prompt": prompt,
            "response": response,
            "embedding": embedding
        }).execute()
    except Exception as e:
        print(f"Error storing user context: {e}")

@app.route("/", methods=["GET"])
def home():
    """
    Default route for the Flask application.
    """
    return jsonify({"message": "Welcome to the AI Backend API. Use the /ask endpoint to interact."})

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Handle user prompt, check context, and generate response using Hugging Face InferenceClient.
    """
    data = request.json
    user_id = data.get("user_id")
    prompt = data.get("prompt")

    if not user_id or not prompt:
        return jsonify({"error": "user_id and prompt are required"}), 400

    # Retrieve all past contexts for the user
    user_contexts = get_user_contexts(user_id)

    # Prepare messages for the model
    messages = []
    if user_contexts:
        # Include past contexts in the conversation
        messages.append({"role": "system", "content": "This is the past context for the conversation."})
        for context in user_contexts:
            messages.append({"role": "assistant", "content": f"Prompt: {context['prompt']}"})
            messages.append({"role": "assistant", "content": f"Response: {context['response']}"})

    # Add the current user prompt
    messages.append({"role": "user", "content": prompt})

    # Call Hugging Face InferenceClient
    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=messages,
            max_tokens=512,
        )
        answer = completion.choices[0].message["content"].strip()

        # Store the new context in the unified table
        store_user_context(user_id, prompt, answer)

        # Log the answer to the console
        print(f"Answer for user {user_id}: {answer}")

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
