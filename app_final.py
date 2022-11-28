import os
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import psycopg2
from fastapi import FastAPI, HTTPException, Depends
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from loguru import logger
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import desc,func
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from schema import PostGet

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("catboost_model_100")
    from_file = CatBoostClassifier()

    from_file.load_model(model_path)
    return from_file

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    i=0
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        print(i)
        chunks.append(chunk_dataframe)
        i+=1
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    )
    #create_features()
    features = batch_load_sql("SELECT * FROM erik_tadevosian_users_features_final")
    return features


def df_public_post():
    SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    public_post_query = "SELECT * FROM public.post_text_df"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    df_public_post = pd.read_sql_query(public_post_query, engine)
    tfidf = TfidfVectorizer(max_features=10000)
    df = pd.DataFrame(tfidf.fit_transform(df_public_post['text']).todense(),
                      columns=tfidf.get_feature_names())
    pca = PCA(n_components=100, random_state=0)
    pca_df = pca.fit_transform(df)
    df_pca = pd.DataFrame(pca_df, columns=['PCA' + str(i + 1) for i in range(100)])
    df_public_post = pd.concat([df_public_post, df_pca], axis=1)
    dum = pd.get_dummies(df_public_post['topic'],drop_first=True)
    df_public_post = pd.concat([df_public_post,dum],axis=1)
    df_public_post = df_public_post.drop('topic',axis=1)
    print('df_public_post done')
    return df_public_post


app = FastAPI()
model = load_models()
print('model done')
features = load_features()
print('features done')
df_post = df_public_post()
print("post done")
@app.get("/post/recommendations/", response_model=List[PostGet])
def get_recommendations(id: int,time: datetime, limit: int = 5):
    user = features.loc[features['user_id']==id,['user_id','gender','age','city','exp_group','user_cluster','Belarus','Cyprus','Estonia','Finland','Kazakhstan','Latvia','Russia','Switzerland','Turkey','Ukraine','iOS']]
    user = user.head(1)
    for col in user:
        df_post[col] = user[col].item()
    X_test = df_post.copy()
    X_test = X_test.drop(['user_id','post_id','text'],axis=1)
    df_post['preds'] = model.predict_proba(X_test)[:, 1]
    top_predictions = df_post.sort_values('preds', ascending=False).head(limit).reset_index(drop=True)
    result = []
    for i in top_predictions.index:
        post_id, text = top_predictions.loc[i, 'post_id'], top_predictions.loc[i, 'text']
        topic_themes = ['covid', 'entertainment', 'movie', 'politics', 'sport', 'tech']
        for theme in topic_themes:
            if top_predictions.loc[i, theme] == 1:
                topic = theme
                break
        test_post = PostGet(**{"id": post_id, "text": text, "topic": topic})
        result.append(test_post)
        #print(result)
    if result is None:
        raise HTTPException(200)
    return result

#get_recommendations(200,5,5)
