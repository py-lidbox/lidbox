#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

common_voice_src=/m/data/speech/common-voice
num_files_to_parse=200

cache_dir=./test-cache
experiment_config=./experiment.test.yaml
parsed_common_voice_dir=./acoustic_data
verbosity='-vv'

function check_prerequisities {
	error=0
	if [ -z $(command -v speechbox) ]; then
		echo "Command 'speechbox' not found, did you install the Python package?"
		error=1
	fi
	if [ ! -d "$common_voice_src" ]; then
		echo "Directory '$common_voice_src' does not exist."
		echo "It should contain directories named by the ISO 639-3 identifier matching the language of each extracted Common Voice dataset."
		error=1
	fi
	if [ $error -ne 0 ]; then
		exit $error
	fi
}

check_prerequisities

# Enable line buffering to flush all output as soon it becomes available
if [ -z $PYTHONUNBUFFERED ]; then
	export PYTHONUNBUFFERED=1
fi

if [ ! -d "$parsed_common_voice_dir" ]; then
	printf "'${parsed_common_voice_dir}' does not exist, parsing all downloaded Common Voice dataset mp3 files\n"
	enabled_labels=$(speechbox system --yaml-get dataset.labels "$experiment_config")
	for label in $enabled_labels; do
		src="${common_voice_src}/${label}"
		dst="${parsed_common_voice_dir}/${label}/wav"
		printf "Parsing from '$src' to '$dst'\n"
		if [ -z "$(find "$src" -type f -name '*.mp3')" ]; then
			printf "Error, '$src' does not contain any mp3 files\n"
			continue
		fi
		speechbox parser --parse common-voice --limit $num_files_to_parse $verbosity "$src" "$dst"
		cp --verbose "${src}/validated.tsv" "${dst}/.."
	done
else
	printf "'${parsed_common_voice_dir}' exists, assuming all required Common Voice datasets have been parsed into the directory.\n"
fi

printf "\nReading all audio files from speech corpus in ${parsed_common_voice_dir}\n"
speechbox dataset $cache_dir $experiment_config \
	$verbosity \
	--src $parsed_common_voice_dir \
	--walk \
	--check \
	--save-state

printf "\nCreating random training-test split by speaker ID\n"
speechbox dataset $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--split by-speaker \
	--check-split by-speaker \
	--save-state

printf "\nComputing total duration of dataset audio files\n"
speechbox dataset $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--get-audio-durations

printf "\nExtracting features\n"
speechbox preprocess $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--extract-features \
	--save-state

printf "\nCounting total amount of features\n"
speechbox preprocess $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--count-features \
	--save-state

printf "\nTraining simple LSTM model\n"
speechbox model $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--train \
	--imbalanced-labels

printf "\nEvaluating model\n"
speechbox model $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--evaluate-test-set loss

printf "\nGenerating confusion matrix from predicting labels for all files in the test set\n"
speechbox model $cache_dir $experiment_config \
	$verbosity \
	--load-state \
	--evaluate-test-set confusion-matrix
