# lidbox

* Command line toolbox for managing spoken language identification (LID) experiments, or any discriminative speech classification.
* Parallel feature extraction using `tf.data.Dataset` with STFT computations on the GPU using the `tf.signal` package.
* Average detection cost (`C_avg`) implemented as a `tf.keras.metrics.Metric`.
* Model training with `tf.keras`, some model examples are available [here](./lidbox/models).
* Expect to do some debugging if (when) stuff does not work.

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
