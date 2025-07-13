import os
from app.models.car import Car
from app.db.session import SessionLocal
import requests
import json
from dotenv import load_dotenv
from collections import defaultdict
from app.models.carembeddings import search_similar_cars
import boto3

load_dotenv()
api_key = os.getenv('AWS_BEARER_TOKEN_BEDROCK')

# Store chat history per user/session
chat_memories = defaultdict(list)

if not api_key:
    raise ValueError("API key not found. Make sure AWS_BEARER_TOKEN_BEDROCK is set.")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

#endpoint = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"

# System prompt for the assistant's behavior
SYSTEM_PROMPT = (
    "You are an expert virtual assistant for a car rental booking platform. "
    "Your role is to help users search for available cars, compare options, provide detailed information about vehicles, assist with booking reservations, manage existing bookings, and answer any questions related to car rentals, pricing, insurance, pick-up/drop-off locations, and payment. "
    "Always be clear, friendly, and concise. Guide the user step-by-step through the booking process, collect all necessary details (such as location, dates, car type, and driver information), and confirm actions before finalizing any reservation or cancellation. "
    "If a user asks about availability, pricing, or policies, provide accurate and up-to-date information. If you do not know the answer, politely let the user know and suggest they contact customer support. Never make up information or fabricate bookings. "
    "Your goal is to make the car rental experience smooth, secure, and user-friendly."
)

def chat_with_bedrock(user_id, user_message):
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    # Retrieve history for this user/session
    history = chat_memories[user_id]
    
    # If this is the first message, prepend the system prompt
    if not history or not history[0].startswith("System:"):
        history.insert(0, f"System: {SYSTEM_PROMPT}")
    
    # Add the new user message to the history
    history.append(f"User: {user_message}")

    # --- Semantic search for relevant cars ---
    similar_cars = search_similar_cars(user_message)
    car_summaries = [
        f"{car.manufacturer} {car.model} ({car.type}), {car.num_seats} seats, {car.color}, ${car.daily_rate}/day"
        for car in similar_cars
    ]
    cars_text = "\n".join(car_summaries)
    if cars_text:
        history.append(f"Assistant: Here are some cars you might like:\n{cars_text}")
    # --- End semantic search ---

    # Prepare the full prompt by joining the history
    prompt = "\n".join(history)
    
    # Prepare the Bedrock payload
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.7,
            "topP": 0.9
        }
    }
    response = client.invoke_model(
        modelId="titan-text-express-v1",  # or your embedding model ID
        body=json.dumps(body),
        contentType="application/json"
    )
    # Make the Bedrock API call
    #response = requests.post(endpoint, headers=headers, data=json.dumps(body))
    response.raise_for_status()
    result = response.json()
    assistant_reply = result['results'][0]['outputText']
    
    # Add the assistant's reply to the history
    history.append(f"Assistant: {assistant_reply}")
    
    return assistant_reply




def get_all_cars():
    session = SessionLocal()
    cars = session.query(Car).all()
    session.close()
    return cars

def add_car(car_data):
    session = SessionLocal()
    car = Car(**car_data)
    session.add(car)
    session.commit()
    session.close()
    return car
