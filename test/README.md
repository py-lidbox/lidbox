# Tests

This is a self-contained experiment example that can be used for testing.

## Data

Speech data used for testing will be parsed from [Common Voice](https://voice.mozilla.org/en/datasets) datasets.
Since this repository does not include the data, it should be downloaded from the Common Voice website.
Each dataset must be extracted into a directory named with the 3-letter [ISO 693-3](https://iso639-3.sil.org/code_tables/639/data) code of the language that is spoken in the dataset.

For example, after downloading the Breton language dataset `br.tar.gz`, it should be extracted into a directory named `bre`, which will looks something like this:
```
/m/data/speech/common-voice/bre
├── clips
│   ├── common_voice_br_17331785.mp3
│   ├── common_voice_br_17331786.mp3
│   ├── common_voice_br_17331787.mp3
│   ├── common_voice_br_17331788.mp3
:   :
│   ├── common_voice_br_18906257.mp3
│   └── common_voice_br_18906258.mp3
├── dev.tsv
├── invalidated.tsv
├── other.tsv
├── test.tsv
├── train.tsv
└── validated.tsv
```

By default, the test script uses the following languages:

* `bre`: Breton, version `br_10h_2019-06-12`
* `est`: Estonian, version `et_12h_2019-06-12`
* `mon`: Mongolian, version `mn_9h_2019-06-12`
* `tur`: Turkish, version `tr_10h_2019-06-12`

The languages that will be used for multi-class classification can be changed in `experiment.test.yaml` under key `dataset.labels`.
