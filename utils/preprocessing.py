import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    """
    Cleans raw text by removing URLs, mentions, special characters, and extra spaces.
    Args:
        text (str): Raw input text.
    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text.lower()

def tokenize_and_pad(texts, tokenizer, max_seq_length):
    """
    Tokenizes and pads a list of texts to a fixed sequence length.
    Args:
        texts (list of str): List of input texts.
        tokenizer: Tokenizer instance (e.g., from HuggingFace or Keras).
        max_seq_length (int): Maximum sequence length for padding.
    Returns:
        np.ndarray: Tokenized and padded sequences.
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_seq_length,
        return_tensors="np"
    )
    return encodings["input_ids"]

def prepare_data(texts, labels, tokenizer, max_seq_length, test_size=0.2, random_state=42):
    """
    Prepares train and test datasets by cleaning, tokenizing, and splitting.
    Args:
        texts (list of str): List of input texts.
        labels (list of int): Corresponding labels.
        tokenizer: Tokenizer instance.
        max_seq_length (int): Maximum sequence length for padding.
        test_size (float): Proportion of data to include in the test split.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Clean the texts
    cleaned_texts = [clean_text(text) for text in texts]

    # Tokenize and pad
    tokenized_texts = tokenize_and_pad(cleaned_texts, tokenizer, max_seq_length)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        tokenized_texts, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def decode_sequences(sequences, tokenizer):
    """
    Decodes sequences back to text using the tokenizer's vocabulary.
    Args:
        sequences (list of list of int): Tokenized sequences.
        tokenizer: Tokenizer instance.
    Returns:
        list of str: Decoded text sequences.
    """
    return [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in sequences]
