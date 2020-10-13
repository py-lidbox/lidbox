# Common Voice language identification example

This example shows how to create a language identification dataset from four [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) datasets (version 2020-06-22 assumed):
* Breton (br)
* Estonian (et)
* Mongolian (mn)
* Turkish (tr)

These datasets were chosen for this example because they do not contain too much data for a simple example (2-10 hours each), yet there should be enough data for applying deep learning.

If you want to experiment with other Common Voice datasets, update variable `datasets` in `scripts/prepare.bash` and key `labels` in `config.yaml`.

## Requirements

* [`bc`](https://www.gnu.org/software/bc)
* [`lame`](https://lame.sourceforge.io)
* [`python3`](https://www.python.org/downloads) 3.7 (due to TensorFlow 2.1)
* [`sox`](http://sox.sourceforge.net)
* [`tar`](https://www.gnu.org/software/tar)
* [`tensorflow`](https://www.tensorflow.org/install) 2.1 or newer

Python 3.8.2 with TensorFlow 2.2.0-rc2 has also been tested.

## Steps

### Prepare

1. Download the datasets from the [Common Voice](https://voice.mozilla.org/en/datasets) website into `./downloads`.
After downloading, the directory should contain the following files:
    * `br.tar.gz`
    * `et.tar.gz`
    * `mn.tar.gz`
    * `tr.tar.gz`

2. Run

        bash scripts/prepare.bash
    This will extract all mp3-files and convert them to 16k mono wav-files, which creates about 4G of data into directory `./data`.
    The script is very inefficient and the amount of file IO latency makes it unusable for larger datasets, but it will do for this small example.
    After the command completes, the mp3-files are no longer needed.
    It is not necessary to delete them, but you can do it if you want to:

        rm -r ./data/??/clips

### Train an x-vector model on log-scale Mel-spectrograms

**Note** there's also a [notebook](./common-voice-4.ipynb) example on how to use the [API](/lidbox/api.py).

3. Run the `lidbox` end-to-end pipeline. The training should take a few minutes if you have a decent GPU, but might require up to 30 minutes on a CPU. Early stopping is used based on the validation loss.

        lidbox e2e -v config.yaml
    You can enable debug mode by setting the environment variable `LIDBOX_DEBUG=true`, but note that this generates a lot of output.
    It also disables most of the parallel execution.

4. Inspect extracted features and training progress in TensorBoard by running:

        tensorboard --samples_per_plugin="images=0,audio=0,text=0" --logdir ./lidbox-cache/xvector/common-voice-4/tensorboard
    Then go to the localhost web address that TensorBoard is using (probably http://localhost:6006).
    Take some time inspecting the data in all the tabs, e.g. look at the Mel filter banks under 'images' and listen to the utterances under 'audio'.
    In my experience, TensorBoard seems to work best on Chrome, ok on Firefox and quite poorly on Safari.

### Train Gaussian Naive Bayes on x-vector embeddings

5. Install the [PLDA package](https://github.com/RaviSoji/plda):

        python3 -m pip install plda@https://github.com/RaviSoji/plda/archive/184d6e39b01363b72080f2752819496cd029f1bd.zip


6. Extract language embeddings from the x-vector model and train Gaussian Naive Bayes from `scikit-learn`:

        lidbox train-embeddings -v config.xvector-NB.yaml
    Some samples of the extracted embeddings/vectors from both the training and test set can be found as PNG images in `./lidbox-cache/naive_bayes/common-voice-4-embeddings/figures`.

If you want to do some other experiments with the embeddings, you can extract them as NumPy arrays with your own [script](./scripts/get_embeddings.py).

### Extra

You can patch the default feature extraction [pipeline](../../lidbox/dataset/pipelines.py) with external scripts.
Here I'm using [`compute_stats.py`](./compute_stats.py) from the current example directory.
It computes VAD decisions on the input audio and then counts how many frames were dropped and how many were kept.

        lidbox utils -v config.yaml --split test --run-script compute_stats.py

### TODO

Examples how to use the dataset steps API.
The CLI is a bit cumbersome.

## Notes

* You can include any kind of dataset by using Kaldi-like metadata files, see contents of `./data/{train,test}` after running step 2.
* If the command line interface is too restricting, take look at the [notebook example](./common-voice-4.ipynb)
for API usage examples.
* If you don't want to use a GPU, you can e.g. prefix all commands with `env CUDA_VISIBLE_DEVICES=-1`.
* Debug mode can be enabled by using the env var `LIDBOX_DEBUG=true`.
* Keep an eye on the memory usage if you are using large feature extraction batches, there's no kill switch for using too large batches.
* If you modify the config file, there's a JSON schema for validation, which can be checked with `lidbox utils -v --validate-config-file config.yaml`. This might give some useful or useless error messages.
