from app.models.car import Car
from app.db.session import SessionLocal

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
