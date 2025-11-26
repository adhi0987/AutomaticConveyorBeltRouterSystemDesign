from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class adminUser(Base):
    __tablename__ = "admin_logins"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False) 
    password = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)

class employeeUser(Base):
    __tablename__ = "employee_logins"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False) 
    password = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)

class CityData(Base):
    __tablename__ = "cities"
    unique_id = Column(Integer, primary_key=True, index=True)
    cityname = Column(String, unique=True, index=True, nullable=False)

class DataPoint(Base):
    __tablename__ = "datapoints"
    id = Column(Integer, primary_key=True, index=True)
    source_city_id = Column(Integer, nullable=False)
    source_city_name = Column(String, nullable=False)
    destination_city_id = Column(Integer, nullable=False)
    destination_city_name = Column(String, nullable=False)
    parcel_type = Column(Integer, nullable=False)
    route_direction = Column(Integer, nullable=False)

