# Common Voice language identification example

This example shows how to create a language identification dataset from four [Mozilla Common Voice](https://voice.mozilla.org/en/datasets) datasets:
* Breton (br)
* Estonian (et)
* Mongolian (mn)
* Turkish (tr)

These datasets were chosen for this example because they do not contain too much data for a simple example (around 10 hours each), yet there should be enough data for applying deep learning.

**Note** that `lidbox` is still quite rough around the edges so you might need to do some debugging if stuff goes wrong.

## Requirements

* [`bc`](https://www.gnu.org/software/bc)
* [`lame`](https://lame.sourceforge.io)
* [`python3`](https://www.python.org/downloads)
* [`sox`](http://sox.sourceforge.net)
* [`tar`](https://www.gnu.org/software/tar)
* [`tensorflow`](https://www.tensorflow.org/install) 2.0 or newer

## Steps

1. Download the datasets from the [Common Voice](https://voice.mozilla.org/en/datasets) website into `./downloads`.
After downloading, the directory should contain the following files:
    * `br.tar.gz`
    * `et.tar.gz`
    * `mn.tar.gz`
    * `tr.tar.gz`

2. Run

        bash scripts/prepare.bash
This will convert all mp3-files to wav-files, which creates about 6G of data into directory `./common-voice-data`.
After the command completes, the mp3-files are no longer needed.
It is not necessary to delete them, but you can do it if you want to:

        rm -r ./common-voice-data/??/clips

3. Run the `lidbox` end-to-end pipeline with e.g. 100 files for a few epochs to check everything is working:

        lidbox e2e train -vvv config.xvector.yaml --file-limit 100 --exhaust-dataset-iterator --debug-dataset
If `lidbox` does not crash after a few epochs you can interrupt training.
You can also use verbosity level 4 (`-vvvv`) for debugging, but this generates a lot of output.

4. Start TensorBoard by running:

        tensorboard --samples_per_plugin="images=0,audio=0,text=0" --logdir ./lidbox-cache/xvector/tensorboard/logs
Then go to the localhost web address that TensorBoard is using (probably http://localhost:6006).
Take some time inspecting the data before running the pipeline on the whole dataset.

5. Next we will train using whole dataset. First, clear the testing cache that contains samples only for 100 files:

        rm -r ./lidbox-cache

6. Train the model:

        lidbox e2e train -vvv config.xvector.yaml
