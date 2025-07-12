from sqlalchemy import Column, Integer, Numeric, SmallInteger, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Car(Base):
    __tablename__='cars'
    __table_args__ = {'schema': 'car_inventory'}
    car_id = Column(Integer,primary_key=True)
    car_number = Column(String(10),nullable=False)
    model = Column(String(50),nullable=False)
    manufacturer = Column(String(50),nullable=False)
    type = Column(String(50),nullable=False)
    num_seats = Column(SmallInteger,nullable=False)
    color = Column(String(20),nullable=False)
    daily_rate = Column(Numeric(10,2),nullable=False)
