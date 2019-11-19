import tensorflow as tf


def loader(input_shape, output_shape, num_units, num_layers=1, narrowing=False, batch_normalize=False):
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
    if batch_normalize:
        layers = []
        for i, lstm in enumerate(lstm_layers, start=1):
            bn = tf.keras.layers.BatchNormalization(name="batchnorm_{}".format(i))
            layers.extend([lstm, bn])
        lstm_layers = layers
    output = tf.keras.layers.Dense(output_shape, activation="softmax", name="output")
    return tf.keras.models.Sequential(lstm_layers + [output])
