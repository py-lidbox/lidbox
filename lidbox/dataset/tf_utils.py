import sys

import tensorflow as tf

import lidbox.features as features
import lidbox.features.audio as audio_features


def tf_print(*args, **kwargs):
    if "summarize" not in kwargs:
        kwargs["summarize"] = -1
    if "output_stream" not in kwargs:
        kwargs["output_stream"] = sys.stdout
    return tf.print(*args, **kwargs)


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
    num_labels = tf.size(labels)
    labels_enum = tf.range(num_labels)
    # Label to int or one past last one if not found
    # TODO slice index out of bounds is probably not a very informative error message
    out_of_bounds_result = tf.cast(num_labels, dtype=tf.int32)
    label2int = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(labels, labels_enum),
        out_of_bounds_result)
    OH = tf.one_hot(labels_enum, num_labels)
    return label2int, OH


def matplotlib_colormap_to_tensor(colormap_key):
    """
    Given a matplotlib colormap name, extract all RGB values from its cmap numpy array into a tf.constant.
    """
    from matplotlib.cm import get_cmap
    from numpy import arange
    cmap = get_cmap(colormap_key)
    rgb = cmap(arange(cmap.N + 1))[:,:3]
    return tf.convert_to_tensor(rgb)


@tf.function
def tensors_to_rgb_images(inputs, colors, size_multiplier=1):
    """
    Map all values of 'inputs' between [0, 1] and then into RGB color indices.
    Gather colors from 'colors' using the indices.
    Based on https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
    """
    tf.debugging.assert_rank(inputs, 3, message="tensors_to_rgb_images expects batches of 2 dimensional tensors with shape [batch, cols, rows].")
    tf.debugging.assert_rank(colors, 2, message="tensors_to_rgb_images expects a colormap of shape [color, component].")
    tf.debugging.assert_equal(tf.shape(colors)[1], 3, message="tensors_to_rgb_images expects an RGB colormap.")
    # Scale features between 0 and 1 to produce a grayscale image
    inputs = features.feature_scaling(inputs, tf.constant(0.0), tf.constant(1.0))
    # Map linear colormap over all grayscale values [0, 1] to produce an RGB image
    indices = tf.cast(tf.math.round(inputs * tf.cast(tf.shape(colors)[0] - 1, tf.float32)), tf.int32)
    tf.debugging.assert_non_negative(indices, message="Negative color indices")
    images = tf.gather(colors, indices, axis=0, batch_dims=0)
    tf.debugging.assert_rank(images, 4, message="Gathering colors failed, output images do not have a channel dimension. Make sure the inputs have a known shape.")
    # Here it is assumed the output images are going to Tensorboard
    images = tf.image.transpose(images)
    images = tf.image.flip_up_down(images)
    # Rows and columns from shape
    old_size = tf.cast(tf.shape(images)[1:3], tf.float32)
    new_size = tf.cast(size_multiplier * old_size, tf.int32)
    images = tf.image.resize(images, new_size)
    tf.debugging.assert_all_finite(images, message="Tensor conversion to RGB images failed, non-finite values in output")
    return images


@tf.function
def count_dim_sizes(ds, element_key=0, ndims=1):
    """
    Given a dataset 'ds' of 'ndims' dimensional tensors at element index 'element_key', accumulate the shape counts of all tensors.
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
    get_shape_at_index = lambda t: tf.shape(t[element_key])
    shapes_ds = ds.map(get_shape_at_index)
    ones = tf.ones(ndims, dtype=tf.int32)
    shape_indices = tf.range(ndims, dtype=tf.int32)
    def accumulate_dim_size_counts(counter, shape):
        enumerated_shape = tf.stack((shape_indices, shape), axis=1)
        return tf.tensor_scatter_nd_add(counter, enumerated_shape, ones)
    max_sizes = shapes_ds.reduce(
        tf.zeros(ndims, dtype=tf.int32),
        lambda acc, shape: tf.math.maximum(acc, shape))
    max_max_size = tf.math.reduce_max(max_sizes)
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


@tf.function
def extract_features(signals, sample_rates, feattype, spec_kwargs, melspec_kwargs, mfcc_kwargs, db_spec_kwargs, feat_scale_kwargs, window_norm_kwargs):
    tf.debugging.assert_rank(signals, 2, message="Input signals for feature extraction must be batches of mono signals without channels, i.e. of shape [B, N] where B is batch size and N number of samples.")
    tf.debugging.assert_equal(sample_rates, [sample_rates[0]], message="Different sample rates in a single batch not supported, all signals in the same batch should have the same sample rate.")
    #TODO batches with different sample rates (probably not worth the effort)
    sample_rate = sample_rates[0]
    X = audio_features.spectrograms(signals, sample_rate, **spec_kwargs)
    tf.debugging.assert_all_finite(X, "spectrogram failed")
    if feattype in ("melspectrogram", "logmelspectrogram", "mfcc"):
        X = audio_features.melspectrograms(X, sample_rate=sample_rate, **melspec_kwargs)
        tf.debugging.assert_all_finite(X, "melspectrogram failed")
        if feattype in ("logmelspectrogram", "mfcc"):
            X = tf.math.log(X + 1e-6)
            tf.debugging.assert_all_finite(X, "logmelspectrogram failed")
            if feattype == "mfcc":
                coef_begin = mfcc_kwargs.get("coef_begin", 1)
                coef_end = mfcc_kwargs.get("coef_end", 13)
                mfccs = tf.signal.mfccs_from_log_mel_spectrograms(X)
                X = mfccs[..., coef_begin:coef_end]
                tf.debugging.assert_all_finite(X, "mfcc failed")
    elif feattype in ("db_spectrogram",):
        X = audio_features.power_to_db(X, **db_spec_kwargs)
        tf.debugging.assert_all_finite(X, "db_spectrogram failed")
    if feat_scale_kwargs:
        X = features.feature_scaling(X, **feat_scale_kwargs)
        tf.debugging.assert_all_finite(X, "feature scaling failed")
    if window_norm_kwargs:
        X = features.window_normalization(X, **window_norm_kwargs)
        tf.debugging.assert_all_finite(X, "window normalization failed")
    return X


#TODO
# for conf in augment_config:
#     # prepare noise augmentation
#     if conf["type"] == "additive_noise":
#         noise_source_dir = conf["noise_source"]
#         conf["noise_source"] = {}
#         with open(os.path.join(noise_source_dir, "id2label")) as f:
#             id2label = dict(l.strip().split() for l in f)
#         label2path = collections.defaultdict(list)
#         with open(os.path.join(noise_source_dir, "id2path")) as f:
#             for noise_id, path in (l.strip().split() for l in f):
#                 label2path[id2label[noise_id]].append((noise_id, path))
#         tmpdir = conf.get("copy_to_tmpdir")
#         if tmpdir:
#             if os.path.isdir(tmpdir):
#                 if verbosity:
#                     print("tmpdir for noise source files given, but it already exists at '{}', so copying will be skipped".format(tmpdir))
#             else:
#                 if verbosity:
#                     print("copying all noise source wavs to '{}'".format(tmpdir))
#                 os.makedirs(tmpdir)
#                 tmp = collections.defaultdict(list)
#                 for noise_type, noise_paths in label2path.items():
#                     for noise_id, path in noise_paths:
#                         new_path = os.path.join(tmpdir, noise_id + ".wav")
#                         shutil.copy2(path, new_path)
#                         if verbosity > 3:
#                             print(" ", path, "->", new_path)
#                         tmp[noise_type].append((noise_id, new_path))
#                 label2path = tmp
#         for noise_type, noise_paths in label2path.items():
#             conf["noise_source"][noise_type] = [path for _, path in noise_paths]
# def chunk_loader(signal, sr, meta, *args):
#     chunk_length = int(sr * chunk_len_seconds)
#     max_pad = int(sr * max_pad_seconds)
#     if signal.size + max_pad < chunk_length:
#         if verbosity:
#             tf_util.tf_print("skipping too short signal (min chunk length is ", chunk_length, " + ", max_pad, " max_pad): length ", signal.size, ", utt ", meta[0], output_stream=sys.stderr, sep='')
#         return
#     uttid = meta[0].decode("utf-8")
#     yield from chunker(signal, target_sr, (uttid, *meta[1:]))
#     rng = np.random.default_rng()
#     for conf in augment_config:
#         if conf["type"] == "random_resampling":
#             # apply naive speed modification by resampling
#             ratio = rng.uniform(conf["range"][0], conf["range"][1])
#             with contextlib.closing(io.BytesIO()) as buf:
#                 soundfile.write(buf, signal, int(ratio * target_sr), format="WAV")
#                 buf.seek(0)
#                 resampled_signal, _ = librosa.core.load(buf, sr=target_sr, mono=True)
#             new_uttid = "{:s}-speed{:.3f}".format(uttid, ratio)
#             new_meta = (new_uttid, *meta[1:])
#             yield from chunker(resampled_signal, target_sr, new_meta)
#         elif conf["type"] == "additive_noise":
#             for noise_type, db_min, db_max in conf["snr_def"]:
#                 noise_signal = np.zeros(0, dtype=signal.dtype)
#                 noise_paths = []
#                 while noise_signal.size < signal.size:
#                     rand_noise_path = rng.choice(conf["noise_source"][noise_type])
#                     noise_paths.append(rand_noise_path)
#                     sig, _ = librosa.core.load(rand_noise_path, sr=target_sr, mono=True)
#                     noise_signal = np.concatenate((noise_signal, sig))
#                 noise_begin = rng.integers(0, noise_signal.size - signal.size, endpoint=True)
#                 noise_signal = noise_signal[noise_begin:noise_begin+signal.size]
#                 snr_db = rng.integers(db_min, db_max, endpoint=True)
#                 clean_and_noise = audio_feat.numpy_snr_mixer(signal, noise_signal, snr_db)[2]
#                 new_uttid = "{:s}-{:s}_snr{:d}".format(uttid, noise_type, snr_db)
#                 if not np.all(np.isfinite(clean_and_noise)):
#                     if verbosity:
#                         tf_util.tf_print("warning: snr_mixer failed, augmented signal ", new_uttid, " has non-finite values and will be skipped. Utterance was ", uttid, ", and chosen noise signals were\n  ", tf.strings.join(noise_paths, separator="\n  "), output_stream=sys.stderr, sep='')
#                     return
#                 new_meta = (new_uttid, *meta[1:])
#                 yield from chunker(clean_and_noise, target_sr, new_meta)
