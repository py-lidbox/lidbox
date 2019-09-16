#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

experiment_config=./config.yaml
parsed_common_voice_dir=./acoustic_data
verbosity='-vvv'

error=0
if [ -z $(command -v speechbox) ]; then
	echo "Command 'speechbox' not found, did you install the Python package?"
	error=1
fi
if [ ! -d "$parsed_common_voice_dir" ]; then
	echo "Directory '$parsed_common_voice_dir' does not exist."
	echo "Run the prepare.bash script first."
	error=1
fi
if [ $error -ne 0 ]; then
	exit $error
fi

echo "Reading all audio files from speech corpus in ${parsed_common_voice_dir}"
speechbox dataset gather \
	$experiment_config \
	$verbosity \
	--walk-dir $parsed_common_voice_dir \
	--check \
	--save-state

echo "Inspecting extracted paths"
speechbox dataset inspect \
	$experiment_config \
	$verbosity \
	--dump-datagroup all

echo "Creating random training-test split by speaker ID"
speechbox dataset split \
	$experiment_config \
	by-speaker \
	$verbosity \
	--random \
	--save-state

echo "Checking split is disjoint"
speechbox dataset split \
	$experiment_config \
	by-speaker \
	$verbosity \
	--check

echo "Computing total duration of dataset audio files"
speechbox dataset inspect \
	$experiment_config \
	$verbosity \
	--get-audio-durations

# echo "Augmenting dataset"
# speechbox dataset augment \
# 	$experiment_config \
# 	$verbosity \
# 	--save-state

# echo "Computing total duration of dataset audio files after augmentation"
# speechbox dataset inspect \
# 	$experiment_config \
# 	$verbosity \
# 	--get-audio-durations

echo "Extracting features"
speechbox features extract \
	$experiment_config \
	$verbosity \
	--save-state

echo "Counting total amount of features"
speechbox features count \
	$experiment_config \
	$verbosity

echo "Training simple LSTM model"
speechbox model train \
	$experiment_config \
	$verbosity
