#!/usr/bin/env sh
set -e
self_dir=$(dirname $0)
err_out=$1
python=$2
output_dir=$3
shift 3
$python $self_dir/spherediar_to_numpy.py $output_dir $@ 2> $err_out
