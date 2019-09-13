#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

cache_dir=./test-cache
experiment_config=./config.yaml
parsed_common_voice_dir=./acoustic_data
verbosity='-vvv'

function check_prerequisities {
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
}
check_prerequisities

printf "\nReading all audio files from speech corpus in ${parsed_common_voice_dir}\n"
speechbox dataset gather \
	$experiment_config \
	$verbosity \
	--walk-dir $parsed_common_voice_dir \
	--check \
	--save-state

printf "\nInspecting extracted paths\n"
speechbox dataset inspect \
	$experiment_config \
	$verbosity \
	--dump-datagroup all

printf "\nCreating random training-test split by speaker ID\n"
speechbox dataset split \
	$experiment_config \
	by-speaker \
	$verbosity \
	--random \
	--save-state

printf "\nChecking split is disjoint\n"
speechbox dataset split \
	$experiment_config \
	by-speaker \
	$verbosity \
	--check

printf "\nComputing total duration of dataset audio files before augmentation\n"
speechbox dataset inspect \
	$experiment_config \
	$verbosity \
	--get-audio-durations

# printf "\nAugmenting dataset\n"
# speechbox dataset augment \
# 	$experiment_config \
# 	$verbosity \
# 	--save-state

# printf "\nComputing total duration of dataset audio files after augmentation\n"
# speechbox dataset inspect \
# 	$experiment_config \
# 	$verbosity \
# 	--get-audio-durations

printf "\nExtracting features\n"
speechbox features extract \
	$experiment_config \
	$verbosity \
	--save-state

printf "\nCounting total amount of features\n"
speechbox features count \
	$experiment_config \
	$verbosity

printf "\nTraining simple LSTM model\n"
speechbox model train \
	$experiment_config \
	$verbosity
