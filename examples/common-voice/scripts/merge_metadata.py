"""
Merges all metadata files created by split_test_set.py for each language into a single training and test set split.
"""
import argparse
import os

def main(src, dst):
    data = {k: {m: {} for m in ("path", "label", "dur")} for k in ("test", "train")}
    langdirs = list(os.scandir(src))
    for langdir in langdirs:
        print(langdir.name)
        for dg in ("test", "train"):
            for meta_key, meta in data[dg].items():
                with open(os.path.join(langdir, dg, "utt2" + meta_key)) as f:
                    for utt, val in (l.strip().split() for l in f):
                        assert utt not in meta, "utterance '{}' already in metadata dict of type '{}'".format(utt, meta_key)
                        meta[utt] = val
    for dg in ("test", "train"):
        uttlist = sorted(data[dg]["path"])
        os.makedirs(os.path.join(dst, dg), exist_ok=True)
        for meta_key, meta in data[dg].items():
            with open(os.path.join(dst, dg, "utt2" + meta_key), "w") as f:
                for utt in uttlist:
                    print(utt, meta[utt], file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    main(args.src, args.dst)
