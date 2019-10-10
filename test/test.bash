#!/usr/bin/env bash
set -e

self_dir=$(dirname $0)
config=${self_dir}/config.yaml
parsed_common_voice_dir=${self_dir}/acoustic_data
verbosity='-vv'
feature_extract_workers=8

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

# echo "Reading all audio files from speech corpus in ${parsed_common_voice_dir}"
# speechbox dataset gather $config $verbosity --walk-dir $parsed_common_voice_dir --check --save-state

# echo "Counting amount of extracted paths"
# speechbox dataset inspect $config $verbosity --dump-datagroup all | wc --lines

# echo "Creating random training-test split by speaker ID"
# speechbox dataset split $config by-speaker $verbosity --random --ratio 0.05 --save-state

# echo "Checking that the split is disjoint"
# speechbox dataset split $config by-speaker $verbosity --check

# echo "Computing total duration of all audio files in each split group"
# speechbox dataset inspect $config $verbosity --get-audio-durations

# echo "Augmenting dataset"
# speechbox dataset augment $config $verbosity --save-state

# echo "Computing total duration after augmentation"
# speechbox dataset inspect $config $verbosity --get-audio-durations

# echo "Extracting features"
# speechbox features extract $config $verbosity --num-workers $feature_extract_workers --save-state

# echo "Counting total amount of features"
# speechbox features count $config $verbosity

echo "Training model"
speechbox model train $config $verbosity
