import tensorflow as tf


def loader(input_shape, output_shape, num_cells=8, dropout=0.5, recurrent_dropout=0.5):
    return tf.keras.models.Sequential([
        tf.keras.layers.LSTM(
            num_cells,
            input_shape=input_shape,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        ),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
