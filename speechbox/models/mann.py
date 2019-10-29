"""
Keras layer wrapper for the memory-augmented neural network implementation by snowkylin:
https://github.com/snowkylin/ntm
"""
import tensorflow as tf

from .ntm.mann_cell import MANNCell


def loader(input_shape, output_shape, num_layers=1, **mann_kwargs):
    mann_1 = tf.keras.layers.RNN(
        MANNCell(output_dim=input_shape[1], **mann_kwargs),
        name="MANN_1",
        batch_input_shape=(mann_kwargs["batch_size"], *input_shape),
        stateful=True,
        return_sequences=num_layers > 1
    )
    mann_layers = [mann_1]
    for i in range(2, num_layers + 1):
        mann_i = tf.keras.layers.RNN(
            MANNCell(output_dim=input_shape[1], **mann_kwargs),
            name="MANN_{}".format(i),
            stateful=True,
            return_sequences=i < num_layers
        )
        mann_layers.append(mann_i)
    output = tf.keras.layers.Dense(output_shape, activation='softmax')
    return tf.keras.models.Sequential(mann_layers + [output])
