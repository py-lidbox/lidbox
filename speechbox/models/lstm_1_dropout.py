import tensorflow as tf


def loader(input_shape, output_shape, num_cells=8, dropout=0.5):
    lstm = tf.keras.layers.LSTM(
        num_cells,
        input_shape=input_shape,
        dropout=dropout
    )
    output = tf.keras.layers.Dense(output_shape, activation='softmax')
    return tf.keras.models.Sequential([lstm, output])
