# Tests

This is a self-contained experiment example that can be used for testing.

## Data

Speech data used for testing will be parsed from [Common Voice](https://voice.mozilla.org/en/datasets) datasets.
Since this repository does not include the data, it should be downloaded from the Common Voice website.
Each dataset must be extracted into a directory named with the 3-letter [ISO 693-3](https://iso639-3.sil.org/code_tables/639/data) code of the language that is spoken in the dataset.

By default, the test script uses the following datasets (because they have reasonably small size):

* `est`: Estonian, version `et_12h_2019-06-12`
* `mon`: Mongolian, version `mn_9h_2019-06-12`
* `nld`: Dutch, version `nl_23h_2019-06-12`
* `tur`: Turkish, version `tr_10h_2019-06-12`

The languages that will be used for multi-class classification can be changed in `config.yaml` under key `dataset.labels`.
