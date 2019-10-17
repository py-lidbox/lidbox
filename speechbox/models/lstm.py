import tensorflow as tf


def loader(input_shape, output_shape, num_units, num_layers=1, narrowing=False):
    lstm_layers = []
    lstm_1 = tf.keras.layers.LSTM(
        num_units,
        name="LSTM_1",
        input_shape=input_shape,
        return_sequences=num_layers > 1
    )
    lstm_layers.append(lstm_1)
    for i in range(2, num_layers + 1):
        if narrowing:
            num_units //= 2
        lstm_i = tf.keras.layers.LSTM(
            num_units,
            name="LSTM_{}".format(i),
            return_sequences=i < num_layers
        )
        lstm_layers.append(lstm_i)
    output = tf.keras.layers.Dense(output_shape, activation='softmax')
    return tf.keras.models.Sequential(lstm_layers + [output])
