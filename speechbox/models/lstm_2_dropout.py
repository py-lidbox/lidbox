import tensorflow as tf


def loader(input_shape, output_shape, num_cells=8, dropout=0.5):
    lstm_1 = tf.keras.layers.LSTM(
        num_cells,
        name="LSTM_1",
        input_shape=input_shape,
        dropout=dropout,
        return_sequences=True,
    )
    lstm_2 = tf.keras.layers.LSTM(
        num_cells,
        name="LSTM_2",
        dropout=dropout
    )
    output = tf.keras.layers.Dense(output_shape, activation='softmax')
    return tf.keras.models.Sequential([lstm_1, lstm_2, output])
