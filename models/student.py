import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam

def build_student_model_lstm(vocab_size, embedding_dim, lstm_units, max_seq_length, num_classes, learning_rate):
    """Builds a student model using LSTM"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length),
        LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_student_model_gru(vocab_size, embedding_dim, gru_units, max_seq_length, num_classes, learning_rate):
    """Builds a student model using GRU"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length),
        GRU(gru_units, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
