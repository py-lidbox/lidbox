# lidbox

* End-to-end spoken language identification (LID) on TensorFlow.
* Parallel feature extraction using `tf.data.Dataset`, with STFT computations on the GPU using the `tf.signal` package.
* Only metadata (e.g. utt2path, utt2label) is fully loaded into memory, rest is done in linear passes  over the dataset with the `tf.data.Dataset` iterator.
* Spectrograms, source audio, and utterance ids can be written into TensorBoard summaries.
* Model training with `tf.keras`, some model examples are available [here](./lidbox/models).
* Average detection cost (`C_avg`) implemented as a `tf.keras.metrics.Metric`.
* You can also try `lidbox` for speaker recognition, since no assumptions will be made of the signal labels. E.g. use utt2speaker as utt2label and see what happens.

[Here](./examples/common-voice/common-voice-4.ipynb) is an example notebook showing `lidbox` in action.

## Why would I want to use this?

* You need a simple, deep learning based speech classification pipeline.
    For example: waveform -> VAD filter -> augment audio data -> serialize all data to a single binary file -> extract log-scale Mel-spectra or MFCC -> use DNN/CNN/LSTM/GRU/attention (etc.) to classify by signal labels
* You have thousands of hours of speech data
* You have a TensorFlow/Keras model that you train on the GPU and want the `tf.data.Dataset` extraction pipeline to also be on the GPU
* You want an end-to-end pipeline that uses TensorFlow 2 as much as possible

## Why would I not want to use this?

* You are happy doing everything with [Kaldi](https://github.com/kaldi-asr/kaldi) or some other toolkits
* You don't want to debug by reading the source code when something goes wrong
* You don't want to install TensorFlow 2 and configure its dependencies (CUDA etc.)
* You need CTC or some other way to train a phoneme recognizer

## Installing

Install TensorFlow 2.1 or 2.2 (both have been tested).

Clone the repo and install `lidbox` as a Python package (note the explicit `./`).
This will install all other required dependencies, but not TensorFlow.
```
git clone --depth 1 https://github.com/matiaslindgren/lidbox.git
pip install ./lidbox
```
Check that the command line entry point is working
```
lidbox -h
```
If not, make sure the `setuptools` entry point scripts (e.g. directory `$HOME/.local/bin`) are on your path.

If everything is working, see [this](./examples/common-voice) for a simple example to get started.

### Note

If you plan on making changes to the code, it is easier to install `lidbox` as a Python package in setuptools develop mode:
```
pip install --editable ./lidbox
```
Then, if you make changes to the code, there's no need to reinstall the package since the changes are reflected immediately.
Just be careful not to make changes when `lidbox` is running, because TensorFlow will use its `autograph` package to convert some of the Python functions to TF graphs, which might fail if the code changes suddenly.

### X-vector embeddings from a trained model for 4 languages

![2-dimensional PCA plot of 400 random x-vectors for 4 Common Voice languages](./examples/common-voice/img/embeddings-PCA-2D.png)
