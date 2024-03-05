from flask import Flask, request, render_template
import re
import nltk
import joblib
import numpy as np
import os
import string
import aspose.words as aw
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained TfidfVectorizer and SVM model
word_vectorizer = joblib.load(open('C:\\Users\\Tharun\\Desktop\\PROJECT ML\\resume_repo\\Resume_classification_app\\word_vectorizer.pkl', 'rb'))
clf_pkl = joblib.load(open('C:\\Users\\Tharun\\Desktop\\PROJECT ML\\resume_repo\\Resume_classification_app\\fmodel.sav', 'rb'))

# Define category mapping
category_mapping = {0: "PeopleSoft Resume", 1: "React JS Developer Resume", 2: "SQL Developer Lightning Insight Resume", 3: "Workday Resume"}

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join('C:\\Users\\Tharun\\Desktop\\PROJECT ML\\resume_repo\\Resume_classification_app\\store', filename)
    uploaded_file.save(file_path)

    # Read file using Aspose Words
    doc = aw.Document(file_path)
    doc_text = doc.get_text().strip()

    # Preprocess text
    doc_text_processed = preprocess(doc_text)
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
    plt.bar(words, freqs)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Common Words')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('static/word_frequency_plot.png')
    plt.close()  # Close the plot to free memory

    # Preprocess and transform text
    tfidf_vector = preprocess_and_transform(doc_text)

    # Predict category using the pre-trained model
    category_id = clf_pkl.predict(tfidf_vector)[0]
    category = category_mapping[category_id]

    return render_template('result.html', category=category)

if __name__ == "__main__":
    app.run(debug=True)
