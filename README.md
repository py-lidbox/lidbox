# lidbox

* Command line toolbox for managing spoken language identification (LID) experiments.
* Parallel feature extraction using `tf.data.Dataset`.
* Model training with `tf.keras`, some model examples are available [here](./lidbox/models).

**Note** that `lidbox` is still quite rough around the edges so you might need to do some debugging if stuff does not work.
Or feel free to just copy paste useful parts into your own program if you want.

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

## Todo

* move as much logic as possible from the CLI code into `lidbox/api.py`
* reduce repetition in the `predict` and `train` commands
* more mutex locks when updating the caches
* simplify the `tf.data.Dataset` e2e pipeline, e.g. always use dicts for iterator elements
