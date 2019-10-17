import tensorflow as tf


def loader(input_shape, output_shape, num_units, num_layers=1, narrowing=False, merge_mode=None):
    blstm_layers = []
    lstm_1 = tf.keras.layers.LSTM(
        num_units,
        return_sequences=num_layers > 1
    )
    blstm_1 = tf.keras.layers.Bidirectional(
        lstm_1,
        merge_mode=merge_mode,
        input_shape=input_shape,
        name="BLSTM_1",
    )
    blstm_layers.append(blstm_1)
    for i in range(2, num_layers + 1):
        if narrowing:
            num_units //= 2
        lstm_i = tf.keras.layers.LSTM(
            num_units,
            return_sequences=i < num_layers
        )
        blstm_i = tf.keras.layers.Bidirectional(
            lstm_i,
            merge_mode=merge_mode,
            name="BLSTM_{}".format(i)
        )
        blstm_layers.append(blstm_i)
    output = tf.keras.layers.Dense(output_shape, activation='softmax')
    return tf.keras.models.Sequential(blstm_layers + [output])
