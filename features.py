import os
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import datetime
#from schema import PostGet
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
    #df_public_post.to_sql('erik_tadevosian_post_features_100', con=engine)
    print('df_public_post done')
    return df_public_post

def df_public_feed():
    SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    df_public_feed = pd.DataFrame()
    #scheme = ['like','view']
    for i in range(2):
        print(i)
        train_query = f"SELECT user_id, (array_agg(post_id ORDER BY timestamp DESC))[1:5] post_id,(array_agg(target ORDER BY timestamp DESC))[1:5] target,(array_agg(timestamp ORDER BY timestamp DESC))[1:5] from public.feed_data WHERE target={i} GROUP BY user_id"
        test = pd.read_sql_query(train_query, engine)
        df_public_feed = pd.concat([test, df_public_feed])
    df_public_feed.columns = pd.Index(['user_id', 'post_id', 'target', 'timestamp'])
    df_public_feed = df_public_feed.explode(['post_id', 'target', 'timestamp'])
    df_public_feed = df_public_feed.sort_values(['user_id'])
    df_public_feed = df_public_feed.astype({'user_id': 'int64', 'post_id': 'int64', 'target': 'int64'})
    print('df_public_feed done')
    return df_public_feed

def df_user_data():
    user_data_query = "SELECT * FROM public.user_data"
    SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    df_user_data = pd.read_sql_query(user_data_query, engine)
    test_user = df_user_data.copy()
    for col in df_user_data.columns:
        #print(col)
        if (df_user_data[col].dtype == 'object'):
            if col == 'city':
                continue
            else:
                dum = pd.get_dummies(df_user_data[col], drop_first=True)
                test_user = pd.concat([test_user, dum], axis=1)
                test_user = test_user.drop(col, axis=1)
    #test_user = test_user["city"]
    test_user["city"] = test_user["city"].astype('category')
    test_user["city"] = test_user["city"].cat.codes
    scaler = StandardScaler()
    df_user_data_test = scaler.fit_transform(test_user.drop('user_id', axis=1))
    kmeans_semg = KMeans(n_clusters=5, n_init=100, random_state=0).fit(df_user_data_test)
    df_user_data['user_cluster'] = kmeans_semg.predict(df_user_data_test)
    #df_user_data['city'] = city
    #df_user_data.to_sql('erik_tadevosian_users_features',con=engine)
    print('df_user_data done')
    return df_user_data

def cat_features(resulting_table):
    cat_columns = [i for i in resulting_table.columns if resulting_table[i].dtype=='object']
    for col in cat_columns:
        #print(col)
        if col!='text':
            if len(resulting_table[col].unique())>15:
                means = resulting_table.groupby(col)['target'].mean()
                noise = np.random.normal(0, 0.1, [resulting_table.shape[0],])
                resulting_table[col] = resulting_table[col].map(means) + noise
            else:
                dum = pd.get_dummies(resulting_table[col],drop_first=True)
                resulting_table = pd.concat([resulting_table,dum],axis=1)
                resulting_table = resulting_table.drop(col,axis=1)
    return resulting_table

def create_features():
    SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    #df_post = df_public_post()
    print('start public feed')
    df_feed = df_public_feed()
    df_user = df_user_data()
    #resulting_table = pd.merge(df_feed, df_post, on='post_id', how='left')
    resulting_table = pd.merge(df_user, df_feed, on='user_id', how='left')
    resulting_table = cat_features(resulting_table)
    user_means = resulting_table.groupby('user_id')['city'].mean()
    df_user['city'] = user_means.values
    for col in df_user:
        if (df_user[col].dtype == 'object'):
            print(col)
            dum = pd.get_dummies(df_user[col], drop_first=True)
            df_user = pd.concat([df_user, dum], axis=1)
            df_user = df_user.drop(col, axis=1)
    df_user_final = df_user[
        ['user_id', 'gender', 'age', 'city', 'exp_group', 'user_cluster', 'Belarus', 'Cyprus', 'Estonia', 'Finland',
         'Kazakhstan', 'Latvia', 'Russia', 'Switzerland', 'Turkey', 'Ukraine', 'iOS']]
    print(df_user_final.info())
    df_user_final.to_sql('erik_tadevosian_users_features_final', con=engine)
    return df_user_final
create_features()