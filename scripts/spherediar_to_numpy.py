import argparse
import json
import os

from SphereDiar import SphereDiar
import librosa.core
import librosa.util
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), "SphereDiar", "models", "SphereSpeaker.hdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("wavpath", type=str, nargs="*")
    args = parser.parse_args()
    spherediar = SphereDiar.SphereDiar(SphereDiar.load_model(model_path))
    path_to_numpyfile = {}
    for wavpath in args.wavpath:
        wav, sample_rate = librosa.core.load(wavpath, sr=None)
        assert sample_rate == 16000, "SphereDiar supports only 16 kHz sample rates"
        if wav.size / sample_rate < 2.0:
            # SphereDiar expects at least 2 second utterances, pad wav to 2 seconds with silence
            wav = librosa.util.fix_length(wav, 2 * sample_rate)
        _ = spherediar.extract_features(wav)
        embedding = spherediar.get_embeddings()
        output_path = os.path.join(args.output_dir, os.path.basename(wavpath).split(".wav")[0])
        output_path += ".npy"
        np.save(output_path, embedding, allow_pickle=False, fix_imports=False)
        path_to_numpyfile[wavpath] = output_path
        spherediar.embeddings_ = []
        spherediar.X_ = []
    print(json.dumps(path_to_numpyfile))
