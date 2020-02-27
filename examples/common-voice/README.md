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
```bash
bash scripts/prepare.bash
```
This will create about 6G of data into directory `./common-voice-data`.
3. Run the `lidbox` end-to-end pipeline with e.g. 100 files for a few epochs to check everything is working:
```
lidbox e2e train -vvv config.xvector.yaml --file-limit 100 --exhaust-dataset-iterator --debug-dataset
```
If `lidbox` does not crash after a few epochs you can interrupt training.
You can also use verbosity level 4 (`-vvvv`) for debugging, but this generates a lot of output.
4. Inspect the extracted data with TensorBoard by running:
```
tensorboard --samples_per_plugin="images=0,audio=0,text=0" --logdir ./lidbox-cache/xvector/tensorboard/logs
```
Then go to the localhost web address that TensorBoard is using (probably http://localhost:6006).
5. Clear the cache before extracting all features:
```
rm -r ./lidbox-cache
```
6. Now extract all features (without training) into the cache:
```
lidbox e2e train -vvv config.xvector.yaml --exhaust-dataset-iterator --skip-training --debug-dataset
```
You can also skip this step but then the first epoch will be much slower since all features are extracted during training.
7. Then train the model:
```
lidbox e2e train -vvv config.xvector.yaml
```
