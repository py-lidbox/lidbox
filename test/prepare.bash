#!/usr/bin/env bash
set -e

self_dir=$(realpath --canonicalize-existing $(dirname $0))
config=${self_dir}/config.yaml
parsed_common_voice_dir=${self_dir}/acoustic_data

common_voice_src=
verbosity='-vv'
# 5 h
num_seconds_to_parse=18000
resampling_rate=16000
# Ignore audio files that are shorter than 4.1 seconds
# 0.1 s extra is to ensure that no file is less than 4 seconds
min_duration_ms=4100
# Normalize volume of all output files
normalize_dBFS="-1.0"

error=0
if [ -z "$(command -v speechbox)" ]; then
	echo "Error: Command 'speechbox' not found, did you install the Python package?"
	error=1
else
	common_voice_src="$(speechbox util --yaml-get dataset.src "$config")"
fi
if [ -z "$(command -v sox)" ]; then
	echo "Error: Command 'sox' not found, cannot parse audio files"
	error=1
elif [ -z "$(sox --help | grep 'AUDIO FILE FORMATS' | grep mp3)" ]; then
	echo "Error: 'sox' does not currently have mp3 format support, you need to install required format libraries. E.g. on ubuntu the package 'libsox-fmt-mp3'."
	error=1
fi
if [ ! -d "$common_voice_src" ]; then
	echo "Error: Corpus directory '$common_voice_src' specified in config file '$config' does not exist."
	echo "It should contain directories named by the ISO 639-3 identifier matching the language of each extracted Common Voice dataset."
	error=1
fi
if [ $error -ne 0 ]; then
	exit $error
fi

echo "Parsing downloaded Common Voice dataset mp3 files as wav files into ${parsed_common_voice_dir}'"
enabled_labels=$(speechbox util --yaml-get dataset.labels "$config")
for label in $enabled_labels; do
	src="${common_voice_src}/${label}"
	dst="${parsed_common_voice_dir}/${label}/wav"
	if [ -z "$(find "$src" -type f -name '*.mp3')" ]; then
		echo "Error, '$src' does not contain any mp3 files"
		continue
	fi
	if [ -d "$dst" ]; then
		echo "Error: '$dst' already exists"
		continue
	fi
	speechbox dataset parse \
		$verbosity \
		--resample-to $resampling_rate \
		--min-duration-ms $min_duration_ms \
		--duration-limit-sec $num_seconds_to_parse \
		--normalize-volume "$normalize_dBFS" \
		common-voice "$src" "$dst"
	echo "copying metadata"
	cp --verbose "${src}/validated.tsv" $(realpath --canonicalize-existing "${dst}/..")
done
