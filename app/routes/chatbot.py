from fastapi import APIRouter
from pydantic import BaseModel

chatbot_router = APIRouter()

class UserQuery(BaseModel):
    question: str

@chatbot_router.post("/ask")
def ask_question(query: UserQuery):
    # Placeholder response
    return {"response": f"Mocked response to: {query.question}"}
