#!/usr/bin/env sh
set -e
cache_dir=./cache
experiment_config=./experiment.example.yaml
speech_corpus_root=./test/data_common_voice

printf "Analyzing all audio files in speech corpus\n\n"
speechbox dataset $cache_dir $experiment_config \
	--verbosity --verbosity \
	--src $speech_corpus_root \
	--check

printf "\nCreating random training-validation-test split\n\n"
speechbox dataset $cache_dir $experiment_config \
	--verbosity --verbosity \
	--src $speech_corpus_root \
	--split by-file \
	--save-state

printf "\nExtracting features\n\n"
speechbox preprocess $cache_dir $experiment_config \
	--verbosity --verbosity \
	--load-state \
	--extract-features \
	--save-state

printf "\nTraining simple LSTM model\n\n"
speechbox train $cache_dir $experiment_config \
	--verbosity --verbosity \
	--load-state \
	--model-id my-simple-lstm \
	--save-model
