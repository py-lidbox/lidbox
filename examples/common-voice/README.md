# Common Voice language identification example

This example shows how to create a language identification dataset from four [Mozilla Common Voice](https://voice.mozilla.org/en/datasets) datasets:
* Breton (br)
* Estonian (et)
* Mongolian (mn)
* Turkish (tr)

These datasets were chosen for this example because they do not contain too much data for a simple example (2-10 hours each), yet there should be enough data for applying deep learning.

## Requirements

* [`bc`](https://www.gnu.org/software/bc)
* [`lame`](https://lame.sourceforge.io)
* [`python3`](https://www.python.org/downloads) 3.7 (due to TensorFlow 2.1)
* [`sox`](http://sox.sourceforge.net)
* [`tar`](https://www.gnu.org/software/tar)
* [`tensorflow`](https://www.tensorflow.org/install) 2.1 or newer

## Steps

1. Download the datasets from the [Common Voice](https://voice.mozilla.org/en/datasets) website into `./downloads`.
After downloading, the directory should contain the following files:
    * `br.tar.gz`
    * `et.tar.gz`
    * `mn.tar.gz`
    * `tr.tar.gz`

2. Run

        bash scripts/prepare.bash
    This will convert all mp3-files to wav-files, which creates about 4G of data into directory `./common-voice-data`.
    After the command completes, the mp3-files are no longer needed.
    It is not necessary to delete them, but you can do it if you want to:

        rm -r ./common-voice-data/??/clips

3. Run the `lidbox` end-to-end pipeline with e.g. 100 files for a few epochs to check everything is working:

        lidbox e2e train -vvv config.xvector.yaml --file-limit 100 --exhaust-dataset-iterator --debug-dataset
    Let it run for e.g. 10 epochs, then interrupt training.
    You can also use verbosity level 4 (`-vvvv`) for debugging, but this generates a lot of output.

4. Next, inspect the extracted features and training progress with TensorBoard by running:

        tensorboard --samples_per_plugin="images=0,audio=0,text=0" --logdir ./lidbox-cache/xvector/tensorboard/logs
    Then go to the localhost web address that TensorBoard is using (probably http://localhost:6006).
    Take some time inspecting the data in all the tabs, e.g. look at the Mel filter banks under 'images' and listen to the utterances under 'audio'.

5. Next we will train using whole dataset. First, clear the testing cache that contains samples only for 100 files:

        rm -r ./lidbox-cache

6. Evaluate only the feature extraction pipeline to fill the features cache:

        lidbox e2e train -vvv config.xvector.yaml --exhaust-dataset-iterator --skip-training
    This step is not really required, since all features could also be extracted during the first epoch during training.

7. Generate TensorBoard data (`--debug-dataset`) and start the training using the extracted features:

        lidbox e2e train -vvv config.xvector.yaml --debug-dataset

8. *TODO: predict test set, compute average detection cost and F1 score*

## Notes

* If you are using a small GPU (less than 4G memory), it might help to prefix the commands that do training with `env TF_FORCE_GPU_ALLOW_GROWTH=true`.
* When extracting features, `lidbox` currently includes original waveforms of each utterance into the features cache in order to make the audio available in TensorBoard. This might create very large caches for large datasets. E.g. I extracted features for approx. 8000 hours of data and it created a 3.2 TiB cache.
