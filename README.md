# lidbox

* Spoken language identification (LId) out of the box using TensorFlow.
* [Models](./lidbox/models) implemented with [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras).
* Metadata handling with [pandas `DataFrames`](https://pandas.pydata.org/docs/reference/frame.html).
* High-performance, parallel preprocessing pipelines with [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)
* Simple spectral and cepstral feature extraction on the GPU with [`tf.signal`](https://www.tensorflow.org/api_docs/python/tf/signal).
* Average detection cost (`C_avg`) [implemented](./lidbox/metrics.py) as a `tf.keras.metrics.Metric` subclass.
* Angular proximity loss [implemented](./lidbox/losses.py) as a `tf.keras.losses.Loss` subclass.

## Why would I want to use this?

* You need a simple, deep learning based speech classification pipeline.
    For example: waveform -> VAD filter -> augment audio data -> serialize all data to a single binary file -> extract log-scale Mel-spectra or MFCC -> use DNN/CNN/LSTM/GRU/attention (etc.) to classify by signal labels
* You want to train a language vector/embedding extractor model (e.g. [x-vector](./lidbox/models/xvector.py)) on large amounts of data.
* You have a TensorFlow/Keras model that you train on the GPU and want the `tf.data.Dataset` extraction pipeline to also be on the GPU
* You want an end-to-end pipeline that uses TensorFlow 2 as much as possible

## Why would I **not** want to use this?

* You are happy doing everything with [Kaldi](https://github.com/kaldi-asr/kaldi) or some other toolkits
* You don't want to debug by reading the source code when something goes wrong
* You don't want to install TensorFlow 2 and configure its dependencies (CUDA etc.)
* You want to train phoneme recognizers or use CTC

## Examples

* [Jupyter notebooks](https://github.com/py-lidbox/examples)
* [Notebooks and all output as HTML](https://py-lidbox.github.io/)


## Installing

Python 3.7 or 3.8 is required.

### Most recent version from PyPI
```
python3 -m pip install lidbox
```

### From source
```
python3 -m pip install https://github.com/py-lidbox/lidbox/archive/master.zip
```

### TensorFlow

TensorFlow 2 is not included in the package requirements because you might want to do custom configuration to get the GPU working etc.

If you don't want to customize anything and instead prefer something that just works for now, the following should be enough:
```
python3 -m pip install tensorflow
```

### Editable install

If you plan on making changes to the code, it is easier to install `lidbox` as a Python package in setuptools develop mode:
```
git clone --depth 1 https://github.com/py-lidbox/lidbox.git
python3 -m pip install --editable ./lidbox
```
Then, if you make changes to the code, there's no need to reinstall the package since the changes are reflected immediately.
Just be careful not to make changes when `lidbox` is running, because TensorFlow will use its `autograph` package to convert some of the Python functions to TF graphs, which might fail if the code changes suddenly.

## Citing `lidbox`

```
@inproceedings{Lindgren2020,
    author={Matias Lindgren and Tommi Jauhiainen and Mikko Kurimo},
    title={{Releasing a Toolkit and Comparing the Performance of Language Embeddings Across Various Spoken Language Identification Datasets}},
    year=2020,
    booktitle={Proc. Interspeech 2020},
    pages={467--471},
    doi={10.21437/Interspeech.2020-2706},
    url={http://dx.doi.org/10.21437/Interspeech.2020-2706}
}
```
