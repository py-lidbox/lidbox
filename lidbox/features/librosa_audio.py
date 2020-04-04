import librosa.core
import tensorflow as tf

# Usage:
# path = /home/it_me/acoustic_data/signal.wav
# signal, sr = tf.numpy_function(audio_feat.py_read_wav, (path,), (tf.float32, tf.int32))
def py_read_wav(path):
    try:
        signal, sr = librosa.core.load(path, sr=None, mono=True)
    except Exception as err:
        tf.print("error: failed to read wav file from", tf.constant(path), "due to exception:", tf.constant(str(err)), summarize=-1)
        signal, sr = tf.zeros([0], tf.float32), 0
    return signal, tf.cast(sr, tf.int32)
