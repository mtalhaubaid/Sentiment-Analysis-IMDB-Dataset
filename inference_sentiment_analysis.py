import numpy as np
import re
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model('best_model_fold_1.h5')


# Define text preprocessing functions

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_stopwords(text):
    stop = set()  # Define your stopwords here
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_urls(text)
    text = remove_stopwords(text)
    return text



# Take user input and preprocess it
user_input = input("Enter a sample movie review: ")
processed_input = denoise_text(user_input)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


sequence = tokenizer.texts_to_sequences([processed_input])
padded_sequence = pad_sequences(sequence, maxlen=100, truncating='post', padding='post')


#  Make a prediction

prediction = model.predict(padded_sequence)
sentiment = "positive" if prediction >= 0.5 else "negative"

print(f"\nThe predicted sentiment for the given review is: {sentiment}")
print(f"Review: {user_input}")

