# lidbox

* End-to-end spoken language identification (LID) on TensorFlow.
* Parallel feature extraction using `tf.data.Dataset`, with STFT computations on the GPU using the `tf.signal` package.
* Only metadata (e.g. utt2path, utt2speaker) is fully loaded into memory, rest is done in linear passes  over the dataset with the `tf.data.Dataset` iterator.
* Model training with `tf.keras`, some model examples are available [here](./lidbox/models).
* Average detection cost (`C_avg`) implemented as a `tf.keras.metrics.Metric`.
* Expect to do some debugging when stuff does not work.

## Quickstart

Install TensorFlow 2.1 or 2.2 (tested with both).

Clone the repo and install `lidbox` as a Python package in setuptools develop mode (`pip install --editable`).
This makes it easier to experiment with the code since there's no need to reinstall the package after making changes.
```
git clone --depth 1 https://github.com/matiaslindgren/lidbox.git
pip install --editable ./lidbox
```
Check that the command line entry point is working
```
lidbox -h
```
If not, make sure the `setuptools` entry point scripts (e.g. directory `$HOME/.local/bin`) are on your path.

If everything is working, see [this](./examples/common-voice) for a simple example to get started.
