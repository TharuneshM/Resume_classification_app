import streamlit as st
import pandas as pd
import re
import nltk
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import aspose.words as aw
import math
from wordcloud import WordCloud, STOPWORDS
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Load pre-trained TfidfVectorizer and SVM model
word_vectorizer = joblib.load(open('word_vectorizer.pkl', 'rb'))
clf_pkl = joblib.load(open('fmodel.sav','rb'))

# Define category mapping
category_mapping = {0: "PeopleSoft Resume", 1: "React JS Developer Resume", 2: "SQL Developer Lightning Insight Resume", 3: "Workday Resume"}
 
# Define Streamlit app
st.title('Resume Classifier')

# Upload resume file
uploaded_file = st.file_uploader('Upload a resume PDF file:', type=['pdf','doc','docx','docs'])

# Preprocessing function
def preprocess(txt):
    txt = txt.lower()
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub('http\S+\s*', ' ', txt)
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('#\S+', '', txt)
    txt = re.sub('@\S+', '  ', txt)
    txt = re.sub('\s+', ' ', txt)
    txt = nltk.tokenize.word_tokenize(txt)
    txt = [w for w in txt if not w in nltk.corpus.stopwords.words('english')]
    return ' '.join(txt)
    
def preprocess_and_transform(text):
    preprocessed_text = preprocess(text)
    tfidf_vector = word_vectorizer.transform([preprocessed_text])
    return tfidf_vector
    
# Process and predict on file
if uploaded_file is not None:
    # Read file using Apose Words
    doc = aw.Document(uploaded_file)
    doc_text = doc.get_text().strip()

    # Preprocess text
    doc_text_processed = preprocess(doc_text)

    # Create a frequency distribution of words
    oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
    totalWords = []
    for word in nltk.word_tokenize(doc_text_processed):
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)

    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(20)

     # Create a bar plot of the most common words
    words = [x[0] for x in mostcommon]
    freqs = [x[1] for x in mostcommon]
    fig, ax = plt.subplots()
    ax.bar(words, freqs)
    ax.set_xticklabels(words, rotation=90)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Most Common Words')
    st.pyplot(fig)

    
    # Generate word cloud
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='black',
                          width=1200,
                          height=1200
                         ).generate(doc_text_processed)

    # Display the word cloud
    st.image(wordcloud.to_array(), use_column_width=True)

    # Preprocess and transform text
    tfidf_vector = preprocess_and_transform(doc_text)

    # Predict category using the pre-trained model
    category_id = clf_pkl.predict(tfidf_vector)[0]

     # Predict probability percentage
    probab= clf_pkl.predict_proba(tfidf_vector)

    #st.write(f"Probability : {probab.values}")
    st.write(f"<span style='font-size:30px;'>Probability : {np.round(probab.max(),3)*100} % </span>", unsafe_allow_html=True)

    #st.write(type(probab))
    category = category_mapping[category_id]

    # Display predicted domain category
    st.write(f"<span style='font-size:30px;'>Predicted Category: {category}</span>", unsafe_allow_html=True)

    #st.write(f'Predicted Category: {category}')

#   streamlit run "C:\Users\Hi\Desktop\DS_Projects\resume\resume_app.py"
