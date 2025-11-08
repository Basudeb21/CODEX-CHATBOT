from sqlalchemy import Column, Integer, BigInteger, Text, Boolean, TIMESTAMP, String
from database import Base
from datetime import datetime

class Client(Base):
    __tablename__ = "client_inqueries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255))
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(50))
    tech_stack = Column(String(255))
    project_description = Column(String(255))
    


class JobSeeker(Base):
    __tablename__ = "job_applicants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255))
    name = Column(String(255))
    email = Column(String(255))
    resume_link = Column(Text)
    skills = Column(Text)
 
