"""
I used ipynb to assess the model I made.

"""
import spacy
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('en_core_web_sm')
# Read in the amazon product reviews
df = pd.read_csv('amazon_product_reviews.csv', sep=',')
# Select reviews to perform sematic analysis on
reviews_data = df['reviews.text']
# Clean data of missing values
clean_data = df.dropna(subset=['reviews.text'])

# Define a function for preprocessing text using spaCy
def preprocess_text(text):
    doc = nlp(text)
    # Lemmatize the tokens and remove stop words
    cleaned_text = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    return cleaned_text

# Define a function for preprocessing text using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"
    
# Apply preprocessing to the reviews text
clean_data['cleaned_reviews'] = clean_data['reviews.text'].apply(preprocess_text)

# Define features and labels
x = clean_data['cleaned_reviews']
y = clean_data['reviews.rating']

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(x_train_vec, y_train)

# Predict sentiment ratings on test set
y_pred = svm_model.predict(x_test_vec)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

my_review_choice = clean_data['reviews.text'][0]
my_next_choice = clean_data['reviews.text'][4]

# Combine the reviews into a list
reviews = [my_review_choice, my_next_choice]

# Vectorize text data
vectorizer = CountVectorizer()
x_vec = vectorizer.fit_transform(reviews)

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(x_vec, x_vec)

# Extract cosine similarity between the two reviews
similarity_score = cosine_sim_matrix[0, 1]
print("Cosine similarity between the two reviews:", similarity_score)