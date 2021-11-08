from google_play_scraper import Sort, reviews, reviews_all
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from PIL import Image
import numpy as np
import pickle
import pandas as pd
import json
import tweepy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#from flasgger import Swagger
import streamlit as st
import joblib
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google_play_scraper import app
from textblob import TextBlob
analyzer = SentimentIntensityAnalyzer()
st.set_option('deprecation.showPyplotGlobalUse', False)
nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


def app_name(app_id):
    result = app(
        app_id,
        lang='en',  # defaults to 'en'
        country='us'  # defaults to 'us'
    )
    return result['title']


def dataset_creation(x):
    result, continuation_token = reviews(
        app_id,
        lang='en',  # defaults to 'en'
        country='us',  # defaults to 'us'
        sort=Sort.MOST_RELEVANT,  # defaults to Sort.MOST_RELEVANT
        count=number_of_reviews,  # defaults to 100
        filter_score_with=x  # defaults to None(means all score)
    )
    df = pd.DataFrame([result[i]['userName']
                      for i in range(len(result))], columns=['Reviewer'])
    df['Reviews'] = [result[i]['content'] for i in range(len(result))]
    df['Reviewer_rating'] = [result[i]['score'] for i in range(len(result))]
    return df


def polarity(text):
    return TextBlob(text).sentiment.polarity


def subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def wordcl():  # wordcloud
    allwords = ''.join([twts for twts in df['Reviews']])
    wordcloud = WordCloud(width=500, height=300, random_state=21,
                          max_font_size=119).generate(allwords)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.pyplot()


def getanalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


def value_coun_graph(x, xlabel, ylabel, title):
    analysis_df['Type of comment'].value_counts()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x.value_counts().plot(kind='bar')
    plt.show()
    st.pyplot()


#a=st.sidebar.radio('Select catogery of Reviews',['reviews with 5 star rating','reviews with 4 star rating','None'])
cat = st.sidebar.selectbox(
    'Catogery', ['View Reviews', 'Sentiment analysis', 'Visuvalization'])
col = st.container()
with col:
    sel_col, disp_col = st.columns(2)
    app_id = sel_col.text_input('Enter app name')
    number_of_reviews = sel_col.number_input('How many reviews you need?')
    rating_ty=sel_col.number_input('Any particular Reviewer Rating type',max_value=5,min_value=None)
    app_name = app_name(app_id)
    disp_col.subheader('Selected app')
    disp_col.write(app_name)
    disp_col.subheader('No of comments Requested')
    disp_col.write(number_of_reviews)
    disp_col.subheader('If nothing was selected It considers all Ratings')
df = dataset_creation(rating_ty)
if cat == 'View Reviews':
    empty_df = pd.DataFrame()
    sel = st.sidebar.selectbox('Select One', [
                               'Requested dataset','Subset of Requested dataset','View reviews based on Rating separetly'])
    # if cat=='Datasets':
    if sel == 'View reviews based on Rating separetly':
        star_1 = st.sidebar.checkbox('Reviews with 1 star rating', value=True)
        star_2 = st.sidebar.checkbox('Reviews with 2 star rating', value=True)
        star_3 = st.sidebar.checkbox('Reviews with 3 star rating', value=True)
        star_4 = st.sidebar.checkbox('Reviews with 4 star rating', value=True)
        star_5 = st.sidebar.checkbox('Reviews with 5 star rating', value=True)
        st.header('Extracts Requested no of Reviews for each rating')
        if star_1:
            df1 = dataset_creation(1)
            empty_df = pd.concat([empty_df, df1])
        if star_2:
            df2 = dataset_creation(2)
            empty_df = pd.concat([empty_df, df2])
        if star_3:
            df3 = dataset_creation(3)
            empty_df = pd.concat([empty_df, df3])
        if star_4:
            df4 = dataset_creation(4)
            # st.dataframe(df4)
            #st.download_button(label='Download 4 star reviews dataset',data=df4.to_csv(),mime='text/csv',file_name='5 star comments.csv')
            empty_df = pd.concat([empty_df, df4])
        if star_5:
            df5 = dataset_creation(5)
            empty_df = pd.concat([empty_df, df5])
            # st.dataframe(df5)
            #st.download_button(label='Download 5 star reviews dataset',data=df5.to_csv(),mime='text/csv',file_name='5 star comments.csv')
        st.dataframe(empty_df)
        if st.download_button(label='Download CSV', data=empty_df.to_csv(), mime='text/csv', file_name='Reviews based on rating.csv'):
            st.balloons()
    if sel == 'Requested dataset':
        st.dataframe(df)
        if st.download_button(label='Download CSV', data=df.to_csv(),mime='text/csv', file_name='Reviews.csv'):
            st.balloons()
                    
    if sel == 'Subset of Requested dataset':
            star_1 = st.sidebar.checkbox('Reviews with 1 star rating', value=True)
            star_2 = st.sidebar.checkbox('Reviews with 2 star rating', value=True)
            star_3 = st.sidebar.checkbox('Reviews with 3 star rating', value=True)
            star_4 = st.sidebar.checkbox('Reviews with 4 star rating', value=True)
            star_5 = st.sidebar.checkbox('Reviews with 5 star rating', value=True)
            st.header('Showing reviews from Requested reviews dataset based on rating:')
            if star_1:
                df1 = df.loc[df['Reviewer_rating'] == 1]
                empty_df = pd.concat([empty_df, df1])
            if star_2:
                df2 = df.loc[df['Reviewer_rating'] == 2]
                empty_df = pd.concat([empty_df, df2])
            if star_3:
                df3 = df.loc[df['Reviewer_rating'] == 3]
                empty_df = pd.concat([empty_df, df3])
            if star_4:
                df4 = df.loc[df['Reviewer_rating'] == 4]
                # st.dataframe(df4)
                #st.download_button(label='Download 4 star reviews dataset',data=df4.to_csv(),mime='text/csv',file_name='5 star comments.csv')
                empty_df = pd.concat([empty_df, df4])
            if star_5:
                df5 = df.loc[df['Reviewer_rating'] == 5]
                empty_df = pd.concat([empty_df, df5])
                # st.dataframe(df5)
                #st.download_button(label='Download 5 star reviews dataset',data=df5.to_csv(),mime='text/csv',file_name='5 star comments.csv')
            st.dataframe(empty_df)
            if st.download_button(label='Download CSV', data=empty_df.to_csv(), mime='text/csv', file_name='Reviews based on rating.csv'):
                st.balloons()
if cat == 'Sentiment analysis':
    menu = ['Dataset with Sentiment score and Subjectivity',
            'Dataset with Comment type']
    sen_cat = st.sidebar.selectbox('Seclect One', menu)
    if sen_cat == 'Dataset with Sentiment score and Subjectivity':
        scores_df = df.copy()
        scores_df['Polarity'] = scores_df['Reviews'].apply(polarity)
        scores_df['Subjectivity'] = scores_df['Reviews'].apply(subjectivity)
        st.dataframe(scores_df)
        if st.download_button(label='Download CSV', data=scores_df.to_csv(), mime='text/csv', file_name='Reviews with Sentiment score.csv'):
            st.balloons()
    if sen_cat == 'Dataset with Comment type':
        scores_df1 = df.copy()
        scores_df1['Polarity'] = scores_df1['Reviews'].apply(polarity)
        scores_df1['Subjectivity'] = scores_df1['Reviews'].apply(subjectivity)
        analysis_df = scores_df1.copy()
        analysis_df['Type of comment'] = analysis_df['Polarity'].apply(
            getanalysis)
        st.dataframe(analysis_df)
        if st.download_button(label='Download CSV', data=analysis_df.to_csv(), mime='text/csv', file_name='Reviews with Final sentiment.csv'):
            st.balloons()
if cat == 'Visuvalization':
    menu = ['WordCloud Image', 'Sentiment vs No of Reviews', 'Rating count']
    vis_cat = st.sidebar.radio('Select', menu)
    if vis_cat == 'WordCloud Image':
        st.header('WordCloud Image of most words used:')
        wordcl()
    if vis_cat == 'Sentiment vs No of Reviews':
        scores_df1 = df.copy()
        scores_df1['Polarity'] = scores_df1['Reviews'].apply(polarity)
        scores_df1['Subjectivity'] = scores_df1['Reviews'].apply(subjectivity)
        analysis_df = scores_df1.copy()
        analysis_df['Type of comment'] = analysis_df['Polarity'].apply(
            getanalysis)
        value_coun_graph(analysis_df['Type of comment'],
                         'Sentiment', 'No of Reviews', 'Sentiment analysis')
    if vis_cat == 'Rating count':
        scores_df1 = df.copy()
        scores_df1['Polarity'] = scores_df1['Reviews'].apply(polarity)
        scores_df1['Subjectivity'] = scores_df1['Reviews'].apply(subjectivity)
        analysis_df = scores_df1.copy()
        analysis_df['Type of comment'] = analysis_df['Polarity'].apply(
            getanalysis)
        value_coun_graph(analysis_df['Reviewer_rating'],
                         'Rating', 'Count', 'Ratings vs count')
