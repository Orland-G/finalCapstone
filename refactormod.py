import spacy
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load('en_core_web_sm')

df = pd.read_csv(r'C:\Users\ogala\OneDrive\Documents\Hyperion Data Analysis\amazon_product_reviews.csv', sep=',')

# Select reviews to perform sematic analysis on
reviews_data = df['reviews.text']

# Clean data of missing values
clean_data = df.dropna(subset=['reviews.text'])

def preprocess_text(text):
    doc = nlp(text)
    # Lemmatize the tokens and remove stop words
    cleaned_text = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    return cleaned_text

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    accuracy = accuracy_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, classification_rep, confusion

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
    
def plot_confusion_matrix(confusion):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Load data
    df = load_data(r'C:\Users\ogala\OneDrive\Documents\Hyperion Data Analysis\amazon_product_reviews.csv', sep=',')

# Clean data
clean_data = df.dropna(subset=['reviews.text'])
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
accuracy, classification_rep, confusion = evaluate_model(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Plot confusion matrix
plot_confusion_matrix(confusion)
print("Confusion Matrix:")
print(confusion)