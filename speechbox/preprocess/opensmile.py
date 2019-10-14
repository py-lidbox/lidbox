"""
Subprocess wrapper over the SMILExtract binary from the openSMILE toolkit.
"""
import os
import numpy as np
import speechbox.system as system
from speechbox.preprocess.transformations import partition_into_sequences


def speech_dataset_to_features(labels, paths, opensmile_conf_path, label_to_index, sequence_length):
    smilextract_cmd = (
        "SMILExtract -nologfile"
        " -configfile " + opensmile_conf_path
        + " -I {path} -O {tmpout}"
    )
    for i, (label, wavpath) in enumerate(zip(labels, paths), start=1):
        tmpout = "{label}-{i}.arff".format(label=label, i=i)
        system.run_command(smilextract_cmd.format(path=wavpath, tmpout=tmpout))
        feats = system.read_arff_features(tmpout, "mfcc")
        os.remove(tmpout)
        onehot = np.zeros(len(label_to_index), dtype=np.float32)
        onehot[label_to_index[label]] = 1.0
        for sequence in partition_into_sequences(feats, sequence_length):
            yield sequence, onehot
