from fastapi import APIRouter
from app.services.car_service import add_car,get_all_cars
from fastapi import FastAPI

app = FastAPI()
router = APIRouter()

@router.get("/cars")
def list_cars():
    return get_all_cars()

@router.post("/cars")
def create_cars(car: dict):
    return add_car(car)

app.include_router(router)