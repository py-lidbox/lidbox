import tensorflow as tf

@tf.function
def count_dim_sizes(ds, ds_element_index=0, ndims=1):
    """
    Given a dataset 'ds' of 'ndims' dimensional tensors at element index 'ds_element_index', accumulate the shape counts of all tensors.
    >>> batch, x, y = 10, 3, 4
    >>> data = tf.random.normal((batch, x, y))
    >>> meta = tf.random.normal((batch, 1))
    >>> ds = tf.data.Dataset.from_tensor_slices((data, meta))
    >>> data_sizes = count_dim_sizes(ds, 0, 2)
    >>> x_size = tf.squeeze(data_sizes[0], 0)
    >>> tf.math.reduce_all(x_size == [batch, x]).numpy()
    True
    >>> y_size = tf.squeeze(data_sizes[1], 0)
    >>> tf.math.reduce_all(y_size == [batch, y]).numpy()
    True
    >>> meta_size = tf.squeeze(count_dim_sizes(ds, 1, 1), [0, 1])
    >>> tf.math.reduce_all(meta_size == [batch, 1]).numpy()
    True
    """
    tf.debugging.assert_greater(ndims, 0)
    get_shape_at_index = lambda *t: tf.shape(t[ds_element_index])
    shapes_ds = ds.map(get_shape_at_index)
    ones = tf.ones(ndims, dtype=tf.int32)
    shape_indices = tf.range(ndims, dtype=tf.int32)
    max_sizes = shapes_ds.reduce(
        tf.zeros(ndims, dtype=tf.int32),
        lambda acc, shape: tf.math.maximum(acc, shape))
    max_max_size = tf.reduce_max(max_sizes)
    @tf.function
    def accumulate_dim_size_counts(counter, shape):
        enumerated_shape = tf.stack((shape_indices, shape), axis=1)
        return tf.tensor_scatter_nd_add(counter, enumerated_shape, ones)
    size_counts = shapes_ds.reduce(
        tf.zeros((ndims, max_max_size + 1), dtype=tf.int32),
        accumulate_dim_size_counts)
    sorted_size_indices = tf.argsort(size_counts, direction="DESCENDING")
    sorted_size_counts = tf.gather(size_counts, sorted_size_indices, batch_dims=1)
    is_nonzero = sorted_size_counts > 0
    return tf.ragged.stack(
        (tf.ragged.boolean_mask(sorted_size_counts, is_nonzero),
         tf.ragged.boolean_mask(sorted_size_indices, is_nonzero)),
        axis=2)

def make_label2onehot(labels):
    """
    >>> labels = tf.constant(["one", "two", "three"], tf.string)
    >>> label2int, OH = make_label2onehot(labels)
    >>> for i in range(3):
    ...     (label2int.lookup(labels[i]).numpy(), tf.math.argmax(OH[i]).numpy())
    (0, 0)
    (1, 1)
    (2, 2)
    """
    labels_enum = tf.range(len(labels))
    # Label to int or one past last one if not found
    # TODO slice index out of bounds is probably not a very informative error message
    label2int = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(labels),
            tf.constant(labels_enum)
        ),
        tf.constant(len(labels), dtype=tf.int32)
    )
    OH = tf.one_hot(labels_enum, len(labels))
    return label2int, OH
