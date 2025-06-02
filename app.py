import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #2c3e50;
            font-size: 36px;
            font-weight: bold;
        }
        .footer {
            margin-top: 50px;
            font-size: 14px;
            color: gray;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.write("Enter the news article text below, and the system will predict whether it's real or fake.")

# Use a text area for better input experience
input_text = st.text_area('✍️ Paste News Content Here:', height=200)

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.markdown("<h4 style='color:red;'>🔴 The News is Fake</h4>", unsafe_allow_html=True)
    else:
        st.markdown("<h4 style='color:green;'>🟢 The News is Real</h4>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        Developed by Jash | Powered by Logistic Regression & NLP 🚀
    </div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
