import tensorflow as tf
from transformers import RobertaTokenizer, TFAutoModel
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

def build_teacher_model(max_seq_length, num_classes, learning_rate):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    teacher_model = TFAutoModel.from_pretrained("roberta-base")

    input_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name='attention_mask')
    bert_output = teacher_model(input_ids, attention_mask=attention_mask).last_hidden_state
    cls_token = bert_output[:, 0, :]  # CLS token
    output = Dense(num_classes, activation='softmax')(cls_token)
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
