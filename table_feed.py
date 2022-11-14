from database import Base,SessionLocal
from sqlalchemy import Column,Integer,String, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from table_user import User
from table_post import Post
class Feed(Base):
    __tablename__ = 'feed_action'
    __table_args__ = {'extend_existing': True}
    user_id = Column(Integer,ForeignKey("user.id"),primary_key = True)
    user = relationship(User)
    post_id = Column(Integer,ForeignKey("post.id"),primary_key = True)
    post = relationship(Post)
    action = Column(String)
    time = Column(TIMESTAMP)