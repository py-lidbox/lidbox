# lidbox

Command line toolbox for managing spoken language identification (LID) experiments.

## Quickstart

```
git clone https://github.com/matiaslindgren/lidbox.git
# install as editable if you want to experiment with the code directly
pip install -e ./lidbox
```
Then install TensorFlow 2.
The `tensorflow` Python package is not specified in `./setup.py` because you might want to do some custom setup on your machine (e.g. to get the GPU working).

See [this](./examples/common-voice) for a simple example to get started.

## Todo

* more mutex locks when updating the caches
* simplify `tf.data.Dataset` pipelines, no spaghetti leaks
* efficient metric implementations, must run in tf graph
