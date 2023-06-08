import nltk
import pandas as pd
import streamlit as st
import re, string
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load dataset
df_teroris = pd.read_csv('indonesian_tweet_about_teroris_(200001-210251).csv')
df_teroris = df_teroris.head(500)
df_teroris.drop_duplicates(inplace=True)
df_teroris = df_teroris.drop(['tweet id','username', 'reference type', 'reference id', 'created at', 'like', 'quote', 'reply', 'retweet', 'tweet url', 'mentions', 'hashtags'], axis=1)

# Text Cleaning
def cleaning(text):
    text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('nan', '', text)
    return text

def preprocess_data(df):
    # Preprocess text
    df['tweet text (clean)'] = df['tweet text'].apply(lambda x: cleaning(x))
    df['tweet text (clean)'] = df['tweet text (clean)'].replace('', np.nan)
    df.dropna(inplace=True)
    
    # Tokenizing tweets
    df['token'] = df['tweet text (clean)'].apply(lambda x: tknzr.tokenize(x))
    
    # Removing stopwords
    stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
    df['token'] = df['token'].apply(lambda x: [w for w in x if not w in stop_words])

    # Labeling tweets
    df['label'] = df['tweet text'].apply(lambda x: label_tweet(x))
    
    return df

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def label_tweet(tweet):
    score = sia.polarity_scores(tweet)
    if score['compound'] >= 0.05:
        return 1  # tweet bernilai positif
    elif score['compound'] <= -0.05:
        return 2  # tweet bernilai negatif
    else:
        return 0  # tweet bernilai netral

# Initialize tokenizer
tknzr = TweetTokenizer()

# Preprocess data
@st.cache
def preprocess_data_cached(df):
    return preprocess_data(df)

# Streamlit App
st.title("Analisis Sentimen dan Pemodelan")
st.header("Analisis Dataset")
st.dataframe(df_teroris)

st.header("Preprocessing dan Tokenisasi")
if st.button("Preprocessing Data"):
    processed_data = preprocess_data_cached(df_teroris)
    st.success("Preprocessing data selesai.")
    st.dataframe(processed_data)

st.header("Analisis Sentimen")
sentiment_text = st.text_input("Masukkan teks untuk analisis sentimen:")
if sentiment_text:
    sentiment_prediction = label_tweet(sentiment_text)
    sentiment_label = "Positif" if sentiment_prediction == 1 else "Negatif" if sentiment_prediction == 2 else "Netral"
    st.success(f"Sentimen: {sentiment_label}")
