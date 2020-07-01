#!/usr/bin/env bash
# 1. Unpacks the downloaded tar.gz files.
# 2. Converts at most 5 hours of randomly chosen Common Voice mp3-clips to 16 kHz mono wav-files, normalized to -3 dBFS.
# 3. Creates a training-test set split such that no speaker is in both sets, with approximately 30 minutes of test data.

# Expanding empty variables is an error
set -u

function print_progress {
	local files=$1
	local seconds=$2
	local hours=$(python3 -c "print(format($seconds/3600, '.2f'))")
	echo "$files files done, $hours hours of data"
}

datasets=(
	br
	et
	mn
	tr
)
# This contains all tar.gz files
downloads_dir=./downloads
# Where to unpack all tars and convert mp3s to wavs
output_dir=./data
cv_artifact=cv-corpus-5-2020-06-22
# Ignore files shorter than 1 second
min_file_dur_sec=1
# Resample all wav-files to this rate before writing
resampling_rate=16k
# Take max 5 hours of mp3 files
max_num_hours_per_dataset=5
# Generate 30 minute test set per language
testset_hours='0.5'
print_progress_step=2000

echo "checking requirements:"
error=0
for cmd in lame sox tar python3 bc; do
	if [ -z "$(command -v $cmd)" ]; then
		echo "  error: required command '$cmd' not found"
		error=1
	else
		echo "  $cmd is $(command -v $cmd)"
	fi
done
if [ ! -d $downloads_dir ]; then
	echo "error: downloads dir '$downloads_dir' does not exist"
	error=1
fi
if [ $error -ne 0 ]; then
	exit $error
fi
echo
echo "using common voice datasets:"
for language in ${datasets[*]}; do
	echo "  $language"
done
echo

# Exit at first error after this
set -e

for language in ${datasets[*]}; do
	tarfile=$downloads_dir/${language}.tar.gz
	echo "unpacking '$tarfile'"
	mkdir -pv $output_dir/$language
	tar zxf $tarfile -C $output_dir/$language
	mv $output_dir/$language/$cv_artifact/$language/* $output_dir/$language
	metadata_tsv=$output_dir/$language/validated.tsv
	mp3_name_list=$(cut -f2 $metadata_tsv | tail -n +2 | shuf)
	if [ -z "$mp3_name_list" ]; then
		echo "error: unable to load list of paths from metadata file at '$metadata_tsv'"
		exit 1
	fi
	num_validated=$(echo $mp3_name_list | wc -w)
	echo "'$language' has $num_validated validated utterances"
	total_sec=0
	total_files=0
	wavs_dir=$output_dir/$language/16k_wavs
	mkdir -pv $wavs_dir
	echo "converting $num_validated mp3 files to wav files into directory '$wavs_dir'"
	for mp3_name in $mp3_name_list; do
		uttid=$(basename -s .mp3 $mp3_name)
		mp3_path=$output_dir/$language/clips/$mp3_name
		if [ ! -f $mp3_path ]; then
			echo "skipping non-existing mp3 file '$mp3_path'"
			continue
		fi
		if [ "$(file --brief $mp3_path)" = empty ]; then
			echo "skipping empty mp3 file '$mp3_path'"
			continue
		fi
		wav_path=$wavs_dir/${uttid}.wav
		if [ -f $wav_path ]; then
			echo "skipping mp3 file '$mp3_path' because wav file '$wav_path' already exists"
			continue
		fi
		lame --quiet --decode $mp3_path - | sox -V1 - $wav_path rate $resampling_rate channels 1 norm -3
		wav_seconds=$(soxi -D $wav_path)
		if [ $(echo "$wav_seconds < $min_file_dur_sec" | bc) -eq 1 ]; then
			echo "skipping too short wav file '$wav_path' of length $wav_seconds sec"
			rm $wav_path
			continue
		fi
		echo $uttid $wav_seconds >> $(dirname $wavs_dir)/utt2dur
		total_files=$(($total_files + 1))
		total_sec=$(echo "$total_sec + $wav_seconds" | bc)
		if [ $(echo "$total_sec > $max_num_hours_per_dataset * 3600" | bc) -eq 1 ]; then
			echo "limit reached at $total_sec seconds, stopping"
			break
		fi
		if [ $(echo "$total_files % $print_progress_step" | bc) -eq 0 ]; then
			print_progress $total_files $total_sec
		fi
	done
	print_progress $total_files $total_sec
	python3 $(dirname $0)/split_test_set.py $output_dir/$language $output_dir/$language $testset_hours
done

echo "all datasets unpacked and converted to wav files"
echo "merging metadata files to a single training set and a single test set"
python3 $(dirname $0)/merge_metadata.py $output_dir $output_dir
echo
echo "all done"
