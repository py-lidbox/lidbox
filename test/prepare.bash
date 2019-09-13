#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

common_voice_src=/m/data/speech/common-voice
num_files_to_parse=6000
resampling_rate=16000
min_duration_ms=2000

experiment_config=./config.yaml
parsed_common_voice_dir=./acoustic_data
verbosity='-v'

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

if [ ! -d "$parsed_common_voice_dir" ]; then
	printf "'${parsed_common_voice_dir}' does not exist, parsing all downloaded Common Voice dataset mp3 files\n"
	enabled_labels=$(speechbox util --yaml-get dataset.labels "$experiment_config")
	for label in $enabled_labels; do
		src="${common_voice_src}/${label}"
		dst="${parsed_common_voice_dir}/${label}/wav"
		if [ -z "$(find "$src" -type f -name '*.mp3')" ]; then
			printf "Error, '$src' does not contain any mp3 files\n"
			continue
		fi
		speechbox dataset parse \
			$verbosity \
			--resample-to $resampling_rate \
			--min-duration-ms $min_duration_ms \
			--limit $num_files_to_parse \
			common-voice "$src" "$dst"
		cp --verbose "${src}/validated.tsv" "${dst}/.."
	done
else
	printf "'${parsed_common_voice_dir}' already exists\n"
fi
