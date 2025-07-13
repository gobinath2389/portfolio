from sqlalchemy import cast
import boto3
import json
from sqlalchemy.orm import Session
from app.models.car import Car
from app.db.session import SessionLocal 
from pgvector.sqlalchemy import Vector
from sqlalchemy import text

def get_embedding(text):
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    body = {
        "inputText": text
    }
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v1",  # or your embedding model ID
        body=json.dumps(body),
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    # Titan returns embeddings in 'embedding' key
    return result['embedding']



 

# get_embedding should be defined as before

def populate_car_embeddings():
    session = SessionLocal()
    try:
        # Fetch cars with NULL embedding
        cars = session.query(Car).filter(Car.embedding == None).all()
        for car in cars:
            text = f"{car.manufacturer} {car.model}, {car.type}, {car.num_seats} seats, {car.color}"
            embedding = get_embedding(text)  # Should return a list of floats
            car.embedding = embedding
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

#populate_car_embeddings()


def search_similar_cars(user_query, max_results=5):
    session = SessionLocal()
    query_embedding = get_embedding(user_query)  # list of floats

    # Format as PostgreSQL vector literal
    embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

    sql = text(f"""
        SELECT * FROM car_inventory.cars
        ORDER BY embedding <=> '{embedding_str}'::vector
        LIMIT :limit
    """)
    result = session.execute(sql, {'limit': max_results})
    cars = result.fetchall()
    session.close()
    return cars

