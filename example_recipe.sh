#!/usr/bin/env sh
set -e

if [ -z $(command -v speechbox) ]; then
	echo "You need to install the speechbox package first"
	exit 1
fi

# Enable line buffering to flush all output as soon it becomes available
if [ -z $PYTHONUNBUFFERED ]; then
	export PYTHONUNBUFFERED=1
fi

cache_dir=./cache/example_experiment
experiment_config=./experiment.example.yaml
speech_corpus_root=./test/data_common_voice
verbosity='-vvv'

printf "Reading all audio files from mini speech corpus in ${speech_corpus_root}\n\n"
speechbox dataset $cache_dir $experiment_config \
	$verbosity \
	--src $speech_corpus_root \
	--walk \
	--check \
	--save-state

# All valid audio files in the speech corpus have now been gathered and their paths and labels saved to the cache directory

printf "\nCreating random training-validation-test split\n\n"
speechbox dataset $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--split by-file \
	--check-split by-file \
	--save-state

# Audio files have now been split into 3 disjoint datagroups: training, validation, and test
# The split, defined by paths, has been saved into the cache directory

printf "\nAugmenting the dataset by transforming audio files\n\n"
speechbox dataset $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--augment \
	--save-state

# The dataset has been augmented by performing SoX transformations defined in the experiment yaml
# The new audio files have been saved into the cache directory and the state has been updated to contain those files

printf "\nChecking dataset integrity\n\n"
speechbox dataset $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--check-integrity

printf "\nExtracting features\n\n"
speechbox preprocess $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--extract-features \
	--save-state

# MFCCs have now been extracted from the audio files for each datagroup, and the features have been saved as TFRecords into the cache directory

printf "\nTraining simple LSTM model\n\n"
speechbox model $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--train

# A simple (read: really bad) keras model has been trained on the features extracted during the previous step and saved into the cache directory

printf "\nEvaluating model\n\n"
speechbox model $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--evaluate-test-set loss
