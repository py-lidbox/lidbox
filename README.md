# lidbox

Command line toolbox for managing spoken language identification (LID) experiments.

## Quickstart

Install TensorFlow 2.

Clone the repo and install `lidbox` as a Python package in setuptools develop mode (`pip install --editable`).
This makes it easier to experiment with the code.
```
git clone --depth 1 https://github.com/matiaslindgren/lidbox.git
pip install --editable ./lidbox
```

See [this](./examples/common-voice) for a simple example to get started.

## Todo

* move as much logic as possible from the CLI code into `lidbox/api.py`
* reduce repetition in the `predict` and `train` commands
* more mutex locks when updating the caches
* simplify `tf.data.Dataset` pipelines, no spaghetti leaks
* efficient metric implementations, must run in tf graph
