from database import Base,SessionLocal
from sqlalchemy import Column,Integer,String, select,func, desc


class User(Base):
    __tablename__ = "user"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key = True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String)
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)
if __name__ == '__main__':
    session = SessionLocal()
    count = func.count("*").label("count")
    result = (session.query(User.country,User.os,count).\
        filter(User.exp_group==3).\
        group_by(User.country,User.os).\
        having(count>100).\
        order_by(desc(count)).all())
    print(result)