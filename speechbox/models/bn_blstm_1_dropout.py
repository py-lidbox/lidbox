import tensorflow as tf


def loader(input_shape, output_shape, num_cells=8, dropout=0.5, merge_mode='concat'):
    lstm = tf.keras.layers.LSTM(num_cells,
        dropout=dropout
    )
    blstm = tf.keras.layers.Bidirectional(lstm,
        name="BLSTM",
        merge_mode=merge_mode,
        input_shape=input_shape
    )
    bn = tf.keras.layers.BatchNormalization()
    output = tf.keras.layers.Dense(output_shape,
        name="softmax",
        activation='softmax'
    )
    return tf.keras.models.Sequential([blstm, bn, output])
