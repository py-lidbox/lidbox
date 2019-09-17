import tensorflow as tf


def loader(input_shape, output_shape, num_cells, num_layers=1, narrowing=False):
    blstm_layers = []
    lstm_1 = tf.keras.layers.LSTM(
        num_cells,
        return_sequences=num_layers > 1
    )
    blstm_1 = tf.keras.layers.Bidirectional(
        lstm_1,
        input_shape=input_shape,
        name="BLSTM_1",
    )
    blstm_layers.append(blstm_1)
    for i in range(2, num_layers + 1):
        if narrowing:
            num_cells //= 2
        lstm_i = tf.keras.layers.LSTM(
            num_cells,
            return_sequences=i < num_layers
        )
        blstm_i = tf.keras.layers.Bidirectional(
            lstm_i,
            name="BLSTM_{}".format(i)
        )
        blstm_layers.append(blstm_i)
    output = tf.keras.layers.Dense(output_shape, activation='softmax')
    return tf.keras.models.Sequential(blstm_layers + [output])
