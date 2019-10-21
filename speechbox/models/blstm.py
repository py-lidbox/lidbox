import tensorflow as tf


def loader(input_shape, output_shape, num_units, num_layers=1, narrowing=False, merge_mode='sum', batch_normalize=False):
    blstm_layers = []
    lstm_1 = tf.keras.layers.LSTM(
        num_units,
        return_sequences=num_layers > 1
    )
    blstm_1 = tf.keras.layers.Bidirectional(
        lstm_1,
        input_shape=input_shape,
        name="BLSTM_1",
        merge_mode=merge_mode,
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
    if batch_normalize:
        layers = []
        for i, blstm in enumerate(blstm_layers, start=1):
            bn = tf.keras.layers.BatchNormalization(name="batchnorm_{}".format(i))
            layers.extend([blstm, bn])
        blstm_layers = layers
    output = tf.keras.layers.Dense(output_shape, activation='softmax')
    return tf.keras.models.Sequential(blstm_layers + [output])
