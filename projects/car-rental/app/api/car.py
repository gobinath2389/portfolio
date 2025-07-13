from pydantic import BaseModel
from fastapi import APIRouter
from app.services.car_service import chat_with_bedrock
from fastapi import FastAPI

app = FastAPI()
router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str
    question: str

@router.post("/chat")
def cars_chat(request: ChatRequest):
    print('question is:', request.question)
    print('user_id is:', request.user_id)
    reply = chat_with_bedrock(request.user_id, request.question)
    return {"response": reply}

app.include_router(router)