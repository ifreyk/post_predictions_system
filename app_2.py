import datetime

import psycopg2
from fastapi import FastAPI, HTTPException, Depends
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from loguru import logger
from table_post import Post
from table_user import User
from table_feed import Feed
from schema import UserGet,PostGet, FeedGet
from database import SessionLocal
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import desc,func

app = FastAPI()


def get_db():
    with SessionLocal() as db:
        return db



@app.get("/user/{id}", response_model=UserGet)
def user_get_id(id: int, db: Session = Depends(get_db)):
        result = db.query(User).filter(User.id==id).first()
        if not result:
            raise HTTPException(404, "user not found")
        else:
            return result
@app.get("/user/{id}/feed",response_model=List[FeedGet])
def user_id_feed(id,limit: int = 10, db: Session = Depends(get_db)):
    result = db.query(Feed).filter(Feed.user_id==id).order_by(Feed.time.desc()).all()[:limit]
    if not result:
        raise HTTPException(200, [])
    else:
        return result

@app.get('/post/{id}',response_model=PostGet)
def post_get_id(id: int,db: Session = Depends(get_db)):
    result = db.query(Post).filter(Post.id==id).first()
    if not result:
        raise HTTPException(404, "post not found")
    else:
        return result


@app.get("/post/{id}/feed",response_model=List[FeedGet])
def post_id_feed(id,limit: int = 10, db: Session = Depends(get_db)):
    result = db.query(Feed).filter(Feed.post_id==id).order_by(Feed.time.desc()).all()[:limit]
    if not result:
        raise HTTPException(200, [])
    else:
        return result

@app.get("/post/recommendations/")
def recomend(limit: int = 10, db: Session = Depends(get_db)):
    result = db.query(Post.id,Post.text,Post.topic).join(Feed).\
        filter(Feed.action=="like").group_by(Post.id).order_by(func.count(Feed.action).desc()).all()[:limit]
    return result
