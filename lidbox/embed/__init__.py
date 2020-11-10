import tensorflow as tf


def extractor2function(extractor):
    extractor.trainable = False
    model_input = extractor.inputs[0]
    extractor_fn = tf.function(
            lambda x: extractor(x, training=False),
            input_signature=[tf.TensorSpec(model_input.shape, model_input.dtype)])
    return extractor_fn.get_concrete_function()
