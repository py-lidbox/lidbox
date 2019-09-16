#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

common_voice_src=
# 10 h
num_seconds_to_parse=36000
resampling_rate=16000
min_duration_ms=2000

experiment_config=./config.yaml
parsed_common_voice_dir=./acoustic_data
verbosity='-v'

error=0
if [ -z "$(command -v speechbox)" ]; then
	echo "Error: Command 'speechbox' not found, did you install the Python package?"
	error=1
else
	common_voice_src="$(speechbox util --yaml-get dataset.src "$experiment_config")"
fi
if [ -z "$(command -v sox)" ]; then
	echo "Error: Command 'sox' not found, cannot parse audio files"
	error=1
elif [ -z "$(sox --help | grep 'AUDIO FILE FORMATS' | grep mp3)" ]; then
	echo "Error: 'sox' does not currently have mp3 format support, you need to install required format libraries. E.g. on ubuntu the package 'libsox-fmt-mp3'."
	error=1
fi
if [ ! -d "$common_voice_src" ]; then
	echo "Error: Corpus directory '$common_voice_src' specified in config file '$experiment_config' does not exist."
	echo "It should contain directories named by the ISO 639-3 identifier matching the language of each extracted Common Voice dataset."
	error=1
fi
if [ -d "$parsed_common_voice_dir" ]; then
	echo "Error: '$parsed_common_voice_dir' already exists"
	error=1
fi
if [ $error -ne 0 ]; then
	exit $error
fi

echo "Parsing downloaded Common Voice dataset mp3 files as wav files into ${parsed_common_voice_dir}'"
enabled_labels=$(speechbox util --yaml-get dataset.labels "$experiment_config")
for label in $enabled_labels; do
	src="${common_voice_src}/${label}"
	dst="${parsed_common_voice_dir}/${label}/wav"
	if [ -z "$(find "$src" -type f -name '*.mp3')" ]; then
		echo "Error, '$src' does not contain any mp3 files"
		continue
	fi
	speechbox dataset parse \
		$verbosity \
		--resample-to $resampling_rate \
		--min-duration-ms $min_duration_ms \
		--duration-limit-sec $num_seconds_to_parse \
		common-voice "$src" "$dst"
	cp --verbose "${src}/validated.tsv" "${dst}/.."
done
