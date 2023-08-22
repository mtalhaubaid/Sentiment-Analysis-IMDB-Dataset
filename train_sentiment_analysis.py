# Import necessary libraries
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import warnings

from bs4 import BeautifulSoup
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Data preprocessing functions


# Check for missing or NaN values in the dataset
missing_values = df.isnull().sum()

print("Missing values in:", missing_values)


def strip_html(text):
    """Remove HTML tags from a text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    """Remove texts between square brackets."""
    return re.sub('\[[^]]*\]', '', text)

def remove_urls(text):
    """Remove URLs from a text."""
    return re.sub(r'http\S+', '', text)

def remove_stopwords(text):
    """Remove stopwords from a text."""
    stop = set()  # Define your stopwords here
    return " ".join([word for word in text.split() if word.strip().lower() not in stop and word.strip().lower().isalpha()])

def denoise_text(text):
    """Apply all preprocessing functions on a text."""
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_urls(text)
    return remove_stopwords(text)

# Preprocess reviews
df['review'] = df['review'].apply(denoise_text)




# Convert sentiment into numerical values
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == "positive" else 0)

# Tokenization and padding
max_len = 100
vocab_size = 10000
embedding_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, truncating='post', padding='post')

# Save the tokenizer for inference
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Model building function
def build_model():
    """Build and compile the CNN model for sentiment analysis."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(32, 3, activation='relu'),
        Dropout(0.7),
        GlobalMaxPooling1D(),
        Dense(8, activation='relu', kernel_regularizer=l2(0.02)),
        Dropout(0.7),
        Dense(1, activation='sigmoid')
    ])
    
    lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# K-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(padded_sequences, df['sentiment'].values)):
    print(f"Fold {fold+1}/{n_splits}")
    
    X_train, X_val = padded_sequences[train_idx], padded_sequences[val_idx]
    y_train, y_val = df['sentiment'].values[train_idx], df['sentiment'].values[val_idx]
    
    model = build_model()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=25, verbose=1)
    checkpoint = ModelCheckpoint(f'best_model_fold_{fold+1}.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop, checkpoint])
    
    # Plot accuracy and loss for each fold
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f'Fold {fold+1} Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'Fold {fold+1} Training and Validation Loss')
    plt.show()

print("Training complete!")
