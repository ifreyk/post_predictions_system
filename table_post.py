from database import Base
from sqlalchemy import Column, Integer, String

class Post(Base):
    __tablename__ = 'post'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)