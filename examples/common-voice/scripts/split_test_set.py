"""
Partitions the dataset in a given directory into training and test sets such that each label has a limited amount of data in the test set.
Common Voice client IDs are used to ensure the training and test sets have different speakers.
"""
import argparse
import csv
import itertools
import os
import random

random.seed(42)

def split_testset(spk2utt, testset_dur_sec):
    test = []
    train = []
    test_dur = 0
    for utts in random.sample(spk2utt, len(spk2utt)):
        if test_dur < testset_dur_sec:
            test.extend(utts)
            test_dur += sum(x['dur_sec'] for x in utts)
        else:
            train.extend(utts)
    assert test_dur >= testset_dur_sec, "not enough test speakers, duration of test set files is {} when {} was requested".format(test_dur, testset_dur_sec)
    return test, train

def group_by_spk(rows):
    get_spk_id = lambda r: r['client_id']
    rows.sort(key=get_spk_id)
    for spk_id, group in itertools.groupby(rows, get_spk_id):
        group = list(group)
        assert all(r['client_id'] == spk_id for r in group)
        yield group

def wavpath2utt(wavpath):
    return os.path.basename(wavpath).split(".wav")[0]

def getdurs(datadir):
    durations = {}
    with open(os.path.join(datadir, "utt2dur")) as f:
        for l in f:
            wavpath, dur = l.split()
            utt = wavpath2utt(wavpath)
            durations[utt] = float(dur)
    return durations

def getrows(datadir, utt2dur):
    utts = set(p.name.split(".wav")[0] for p in os.scandir(os.path.join(datadir, "16k_wavs")))
    with open(os.path.join(datadir, "validated.tsv"), encoding="utf-8") as tsv:
        for row in csv.DictReader(tsv, delimiter='\t'):
            utt = row['path'].split('.mp3')[0]
            if utt in utts:
                row['wavpath'] = os.path.join(datadir, "16k_wavs", utt + '.wav')
                row['dur_sec'] = utt2dur[utt]
                yield row

def is_disjoint(testset, trainset):
    ok = True
    test_ids = set(wavpath2utt(u['wavpath']) for u in testset)
    train_ids = set(wavpath2utt(u['wavpath']) for u in trainset)
    intersection = test_ids & train_ids
    if intersection:
        print("{} utterance ids in both sets".format(len(intersection)))
        ok = False
    test_ids = set(u['client_id'] for u in testset)
    train_ids = set(u['client_id'] for u in trainset)
    intersection = test_ids & train_ids
    if intersection:
        print("{} speaker ids in both sets".format(len(intersection)))
        ok = False
    return ok

def main(src, dst, testset_hours):
    all_data = {}
    for dg in ("train", "test"):
        os.makedirs(os.path.join(dst, dg), exist_ok=True)
        all_data[dg] = {"utt2path": {}, "utt2label": {}, "utt2dur": {}}
    label = os.path.basename(src)
    print("creating training-test split from files in '{}'".format(src))
    print("using language code '{}'".format(label))
    utt2dur = getdurs(src)
    utts = list(getrows(src, utt2dur))
    print(len(utts), "utterances")
    spk2utt = list(group_by_spk(utts))
    print(len(spk2utt), "speakers")
    spk2utt.sort(key=lambda g: len(g))
    testset, trainset = split_testset(spk2utt, testset_hours*3600)
    # disjointness sanity check
    assert is_disjoint(testset, trainset), "failed train-test split"
    print("num utterances:")
    print("test {}, train {}, ratio {:.3f}".format(len(testset), len(trainset), len(testset)/len(trainset)))
    print("total durations in hours:")
    test_hours = sum(utt2dur[wavpath2utt(u['wavpath'])] for u in testset) / 3600
    train_hours = sum(utt2dur[wavpath2utt(u['wavpath'])] for u in trainset) / 3600
    print("test {:.1f} train {:.1f} ratio {:.3f}".format(test_hours, train_hours, test_hours/train_hours))
    print()
    for dg, utterances in (("train", trainset), ("test", testset)):
        for u in utterances:
            utt = wavpath2utt(u['wavpath'])
            dur = utt2dur[utt]
            for p in ("path", "label", "dur"):
                assert utt not in all_data[dg]["utt2" + p], utt + ' utt2' + p
            all_data[dg]["utt2path"][utt] = u["wavpath"]
            all_data[dg]["utt2label"][utt] = label
            all_data[dg]["utt2dur"][utt] = dur
    for dg in ("train", "test"):
        with open(os.path.join(dst, dg, "utt2path"), "w") as fp,\
                open(os.path.join(dst, dg, "utt2label"), "w") as fl,\
                open(os.path.join(dst, dg, "utt2dur"), "w") as fd:
            for u, p in all_data[dg]["utt2path"].items():
                print(u, p, file=fp)
                print(u, all_data[dg]["utt2label"][u], file=fl)
                print(u, all_data[dg]["utt2dur"][u], file=fd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("dst", type=str)
    parser.add_argument("testset_hours", type=float)
    args = parser.parse_args()
    main(args.src, args.dst, args.testset_hours)
