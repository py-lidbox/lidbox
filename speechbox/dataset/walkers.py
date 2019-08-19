"""
Iterator classes for extracting paths, labels, and other metadata from speech corpora and datasets.
"""
import collections
import csv
import os
import re

from speechbox.system import read_wavfile, md5sum, get_audio_type
from . import DatasetRecursionError


class SpeechDatasetWalker:
    """
    Instances of this class are iterable, yielding (label, wavpath) pairs for every file in some dataset, given the root directory of the dataset.
    The tree structure of a particular dataset is defined in the self.label_definitions dict in a subclass of this class.
    """
    # Metadata for each label, should be defined in subclasses inheriting this class
    default_label_definitions = collections.OrderedDict()

    def __init__(self, dataset_root=None, paths=None, labels=None, checksums=None, sample_frequency=None, enabled_labels=None):
        if enabled_labels is None:
            enabled_labels = []
        # Where to start an os.walk from (unless paths and labels explicitly given)
        self.dataset_root = dataset_root
        # If not None, an integer denoting the expected sample frequency/rate in Hz
        self.sample_frequency = sample_frequency
        # Label to speaker id mapping
        self.ignored_speaker_ids_by_label = {}
        # Select enabled labels
        self.label_definitions = collections.OrderedDict()
        for label in enabled_labels:
            self.label_definitions[label] = self.default_label_definitions[label].copy()
        for label, definition in self.label_definitions.items():
            definition["sample_dirs"] = [self.join_root(*paths) for paths in self.dataset_tree[label]]
        if any(l is not None for l in [paths, labels, checksums]):
            assert all(l is not None for l in [paths, labels, checksums])
            # Iterate over given paths instead of searching for all wavs from sample_dirs
            for label_def in self.label_definitions.values():
                label_def["sample_dirs"] = []
                label_def["sample_files"] = []
                label_def["sample_file_checksums"] = []
            # Set all given wavpaths
            for label, path, checksum in zip(labels, paths, checksums):
                self.label_definitions[label]["sample_files"].append(path)
                self.label_definitions[label]["sample_file_checksums"].append(checksum)

    def join_root(self, *paths):
        return os.path.join(self.dataset_root, *paths)

    def load(self, wavpath):
        return read_wavfile(wavpath, sr=self.sample_frequency)

    def get_file_id(self, wavpath):
        return os.path.basename(wavpath).split(".wav")[0]

    def make_label_to_index_dict(self):
        return {label: i for i, label in enumerate(sorted(self.label_definitions))}

    # Different datasets and corpuses denote different speakers in different ways.
    # Implement this method in a subclass for a specific corpus.
    # See e.g. OGIWalker.parse_speaker_id for an example when the speaker id can be deduced from the audio file path.
    # def parse_speaker_id(self, wavpath):
        # pass

    def count_files_per_speaker_by_label(self):
        c = {label: collections.Counter() for label in self.label_definitions}
        for label, path, _ in iter(self):
            speaker_id = self.parse_speaker_id(path)
            c[label][speaker_id] += 1
        return c

    def speakers_per_label(self):
        counts = self.count_files_per_speaker_by_label()
        return {label: len(files_per_speaker) for label, files_per_speaker in counts.items()}

    def speaker_ids_by_label(self):
        counts = self.count_files_per_speaker_by_label()
        return {label: sorted(files_per_speaker.keys()) for label, files_per_speaker in counts.items()}

    def set_speaker_filter(self, speaker_ids_by_label):
        """
        Set wavpath filter such that only files matching the speaker id in the given dict will be yielded from subsequent calls to self.walk().
        """
        self.ignored_speaker_ids_by_label = {label: set(ids) for label, ids in speaker_ids_by_label.items()}

    def speaker_id_is_ignored(self, wavpath, label):
        if label not in self.ignored_speaker_ids_by_label:
            return False
        return self.parse_speaker_id(wavpath) not in self.ignored_speaker_ids_by_label[label]

    def language_label_to_bcp47(self, label):
        """Language label mapping to BCP-47 identifiers.
        Specification: https://tools.ietf.org/html/bcp47
        """
        raise NotImplementedError

    # This function is inherently very messy since it is used to parse clean versions out of very messy speech corpus directories.
    # If there's some inefficient checks that should be run for every file when the corpus is traversed the first time, those checks belong in this function.
    def walk(self, check_duplicates=False, check_read=False, followlinks=True, verbosity=0):
        """
        Walk over all files in the dataset and yield (label, filepath, md5sum) pairs.
        Reads the contents of all audio files in all directories but does not write anything.
        Only files ending with .wav, containing valid WAVE headers will be returned.

        check_duplicates: an MD5 hash will be computed for every file, and if more than 1 files with matching hashes are found, all subsequent files with matching hashes are skipped.
        check_read: every file will try to be opened and its contents checked.
        followlinks: also walk down symbolic links.
        verbosity: from 0 up
        """
        file_extensions=("wav",)
        duplicates = collections.defaultdict(list)
        invalid_files = []
        num_walked = 0
        def audiofile_ok(wavpath, label):
            # First perform validity checks to make sure wavpath contains an audio file
            if not os.path.exists(wavpath):
                if verbosity:
                    print("Warning: {} was supposed to yield file '{}' but it does not exist".format(repr(self), wavpath))
                return False
            if check_read:
                # First try to read wavpath as a wav-file
                wav, srate = read_wavfile(wavpath, sr=None)
                if verbosity and self.sample_frequency and self.sample_frequency != srate:
                    print("Warning: Expected sampling rate set to {} but audio file seems to have native rate {}:".format(self.sample_frequency, srate))
                    print(" ", wavpath)
                # If the read failed, stop checking this file
                if wav is None:
                    invalid_files.append(wavpath)
                    if verbosity > 2:
                        print("Warning: invalid/empty/corrupted audio file:")
                        print(" ", wavpath)
                    return False
                # If the read succeeded, check that the audio file extension matches the contents
                audio_type = get_audio_type(wavpath)
                if verbosity and not (audio_type and wavpath.endswith(audio_type)):
                    print("Warning: file extension does not match contents of type '{}':".format(audio_type))
                    print(" ", wavpath)
            if check_duplicates:
                content_hash = md5sum(wavpath)
                duplicates[content_hash].append(wavpath)
                if len(duplicates[content_hash]) > 1:
                    if verbosity > 2:
                        print("Warning: different files but contents are exactly equal:")
                        for other_wavpath in duplicates[content_hash]:
                            print(" ", other_wavpath)
                    return False
            # Validity checks done, now check the filters for this audio file
            if self.speaker_id_is_ignored(wavpath, label):
                return False
            if not any(wavpath.endswith(ext) for ext in file_extensions):
                return False
            return True
        if verbosity > 1:
            print("Starting walk with walker:", str(self))
        for label in sorted(self.label_definitions.keys()):
            # First walk over all files in all directories specified to contain audio files labeled 'label'
            sample_dirs = self.label_definitions[label].get("sample_dirs", [])
            if verbosity > 2:
                if sample_dirs:
                    print("Label '{}' has {} directories that will now be fully traversed in search for audio files".format(label, len(sample_dirs)))
                else:
                    print("Label '{}' has no directories specified that should be traversed".format(label))
            for sample_dir in sample_dirs:
                seen_directories = set()
                for parent, _, files in os.walk(sample_dir, followlinks=followlinks):
                    this_dir = self.join_root(parent)
                    if this_dir in seen_directories:
                        raise DatasetRecursionError(this_dir + " already traversed at least once. Does the dataset directory contain symbolic links pointing to parent directories?")
                    seen_directories.add(this_dir)
                    for f in files:
                        num_walked += 1
                        wavpath = self.join_root(parent, f)
                        if audiofile_ok(wavpath, label):
                            yield label, wavpath, md5sum(wavpath)
            # Then yield all directly specified wavpaths
            sample_files = self.label_definitions[label].get("sample_files", [])
            if verbosity > 2 and sample_files:
                print("Label '{}' has {} paths to audio files pre-defined, they will now be returned one by one".format(label, len(sample_files)))
            for wavpath in sample_files:
                num_walked += 1
                if audiofile_ok(wavpath, label):
                    yield label, wavpath, md5sum(wavpath)
        if verbosity > 1:
            print("Walk finished by walker:", str(self))
        if verbosity:
            print("Found {} audio files in total".format(num_walked))
            if check_duplicates:
                duplicates = [paths for paths in duplicates.values() if len(paths) > 1]
                if duplicates:
                    print("Found {} duplicate audio files".format(len(duplicates)))
                    if verbosity > 1:
                        print("Groups of files that all have the same MD5 hash of their contents:")
                        print("-- Duplicate file groups begin --")
                        for paths in duplicates:
                            for p in paths:
                                print(p)
                            print()
                        print("-- Duplicate file groups end --")
            if check_read and invalid_files:
                print("Found {} invalid/empty/corrupted audio files".format(len(invalid_files)))
                if verbosity > 1:
                    print("-- Invalid files begin --")
                    for f in invalid_files:
                        print(f)
                    print("-- Invalid files end --")

    def __iter__(self):
        return self.walk()

    def __repr__(self):
        return ("<{}: root={}, num_labels={}>"
                .format(self.__class__.__name__, self.dataset_root, len(self.label_definitions.keys())))

    def __str__(self):
        return ("{} starting at directory '{}', walking over data containing {} different labels"
                .format(self.__class__.__name__, self.dataset_root, len(self.label_definitions.keys())))


class OGIWalker(SpeechDatasetWalker):
    """
    Walker for corpus:
    Cole, Ronald, and Yeshwant Muthusamy. OGI Multilanguage Corpus LDC94S17. Web Download. Philadelphia: Linguistic Data Consortium, 1994.
    Available from https://catalog.ldc.upenn.edu.
    """
    dataset_tree = {
        "cmn": [["wav", "mandarin"]],
        "deu": [["wav", "german"]],
        "eng": [["wav", "english"]],
        "fas": [["wav", "farsi"]],
        "fra": [["wav", "french"]],
        "hin": [["wav", "hindi"]],
        "jpn": [["wav", "japanese"]],
        "kor": [["wav", "korean"]],
        "spa": [["wav", "spanish"]],
        "tam": [["wav", "tamil"]],
        "vie": [["wav", "vietnam"]],
    }
    default_label_definitions = collections.OrderedDict([
        ("cmn", {"name": "Mandarin Chinese"}),
        ("deu", {"name": "German"}),
        ("eng", {"name": "English"}),
        ("fas", {"name": "Persian/Farsi"}),
        ("fra", {"name": "French"}),
        ("hin", {"name": "Hindi"}),
        ("jpn", {"name": "Japanese"}),
        ("kor", {"name": "Korean"}),
        ("spa", {"name": "Spanish"}),
        ("tam", {"name": "Tamil"}),
        ("vie", {"name": "Vietnamese"}),
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def parse_speaker_id(path):
        """
        All filenames should be of form <langcode><callnumber><type>.wav,
        where lengths are:
            langcode: 2
            callnumber: 3
            type: 3
        """
        return os.path.basename(path)[:5]


class OGIWalker2(SpeechDatasetWalker):
    """Legacy thing"""
    dataset_tree = {
        "eng": [
            ("cd01", "speech", "english")
        ],
        "hin": [
            ("cd02", "speech", "hindi"),
            ("cd03", "speech", "hindi")
        ],
        "jpn": [
            ("cd02", "speech", "japanese"),
            ("cd03", "speech", "japanese")
        ],
        "kor": [
            ("cd02", "speech", "korean"),
            ("cd03", "speech", "korean")
        ],
        "cmn": [
            ("cd02", "speech", "mandarin"),
            ("cd03", "speech", "mandarin")
        ],
        "spa": [
            ("cd02", "speech", "spanish"),
            ("cd03", "speech", "spanish"),
            ("cd04", "speech", "spanish")
        ],
        "tam": [
            ("cd04", "speech", "tamil")
        ],
        "vie": [
            ("cd04", "speech", "vietnamese")
        ],
    }
    default_label_definitions = collections.OrderedDict({
        "eng": {"name": "English"},
        "hin": {"name": "Hindi"},
        "jpn": {"name": "Japanese"},
        "kor": {"name": "Korean"},
        "cmn": {"name": "Mandarin Chinese"},
        "spa": {"name": "Spanish"},
        "tam": {"name": "Tamil"},
        "vie": {"name": "Vietnamese"},
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bcp47_mappings = {
            "cmn": "zh", # Chinese, Mandarin (Simplified, China)
            "eng": "en-US",
            "hin": "hi-IN",
            "jpn": "ja-JP",
            "kor": "ko-KR",
            "spa": "es-ES", # Spanish (Spain)
            "tam": "ta-IN", # Tamil (India)
            "vie": "vi-VN",
        }

    @staticmethod
    def parse_speaker_id(path):
        # We assume all wav-files follow a pattern
        #   '..call-[[:digit:]]*-',
        # where the digit between two hyphens is the speaker id.
        # This can be verified by running the following shell command in the ogi dataset dir:
        #   find ogi_multilang_dir -name '*.wav' | grep --invert-match --basic-regexp '..call-[[:digit:]]*-.*wav$'
        return os.path.basename(path).split("call-")[1].split("-")[0]

    def get_phone_segmentation_path(self, wavpath):
        head, tail = wavpath.split("ogi_multilang")
        tail = re.sub(
            os.path.join(r"cd0\d", "speech"),
            os.path.join("cd01", "labels"),
            tail
        )
        tail = tail.replace(".wav", ".ptlola")
        return head + "ogi_multilang" + tail

    def parse_phoneme_segmentation(self, segfile):
        ms_per_frame = 0.0
        with open(segfile) as f:
            lines = iter(f)
            # Header
            for line in lines:
                line = line.strip()
                if line.startswith("MillisecondsPerFrame"):
                    ms_per_frame = float(line.split(":")[-1].strip())
                if "END OF HEADER" in line:
                    break
            for line in lines:
                line = line.strip().split(' ')
                start, end, phoneme = int(line[0]), int(line[1]), line[2]
                yield start, end, phoneme

    def phone_segmentation_to_words(self, phoneseg):
        word_boundaries = {'.pau'}
        # Word is a list of phonemes bounded by two word-boundaries
        word = []
        for _, _, phoneme in phoneseg:
            if phoneme.startswith('.'):
                # A non-phoneme comment starts with a dot
                # See labeling.pdf found in cd01/docs of the ogi dataset for details
                if phoneme in word_boundaries and word:
                    # Word boundary found, finalize new word
                    yield word
                    word = []
                else:
                    # Do not include non-phonemes
                    pass
            else:
                # Add phoneme to word
                word.append(phoneme)
        if word:
            yield word

    def phone_segmentation_for_wavpath(self, wavpath):
        segpath = self.get_phone_segmentation_path(wavpath)
        if not os.path.exists(segpath):
            return
        return self.phone_segmentation_to_words(self.parse_phoneme_segmentation(segpath))

    def language_label_to_bcp47(self, label):
        return self.bcp47_mappings[label]


class VarDial2017Walker(SpeechDatasetWalker):
    """
    Walker for the Vardial 2017 Arabic dialect recognition challenge.
    The dataset division is assumed to be:
        training:   train.vardial2017
        validation: dev.vardial2017
        test:       test.MGB3
    Available from:
    https://github.com/qcri/dialectID/tree/ee6e7e7ca84098eda75cda61892a0ccbd3d8d0bd/data
    """
    dataset_tree = {
        "egy": [["wav", "EGY"]],
        "glf": [["wav", "GLF"]],
        "lav": [["wav", "LAV"]],
        "msa": [["wav", "MSA"]],
        "nor": [["wav", "NOR"]],
    }
    datagroup_patterns = (
        ("training", "train.vardial2017"),
        ("validation", "dev.vardial2017"),
        ("test", "test.MGB3"),
    )
    default_label_definitions = collections.OrderedDict([
        ("egy", {"name": "Egyptian Arabic"}),
        ("glf", {"name": "Gulf Arabic"}),
        ("lav", {"name": "Levantine Arabic"}),
        ("msa", {"name": "Modern Standard Arabic"}),
        ("nor", {"name": "North African Arabic"}),
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def parse_datagroup(cls, wavpath):
        wavpath = os.path.abspath(wavpath)
        directories = set(d for d in os.path.dirname(wavpath).split(os.path.sep) if d)
        for datagroup_key, pattern in cls.datagroup_patterns:
            if pattern in directories:
                return datagroup_key

# Prepend all datagroup directories to all dataset tree directories
for label, dirs in VarDial2017Walker.dataset_tree.items():
    base = dirs.pop()
    dirs.extend([[p] + base for _, p in VarDial2017Walker.datagroup_patterns])


class CommonVoiceWalker(SpeechDatasetWalker):
    """
    Walker for wav-files extracted from mp3-files of a Common Voice dataset.
    Data source: https://voice.mozilla.org/en
    The walker assumes all mp3-files from the 'clips' directory have been extracted into a 'wav' directory next to the 'clips' directory,
    i.e. in the same directory as the tsv metadata files.
    It is also assumed that the user has extracted each dataset into a directory named by the three letter ISO 639-3 identifier of the dataset language.
    """
    dataset_tree = {}
    default_label_definitions = collections.OrderedDict([
        ("bre", {"name": "Breton"}),
        ("cat", {"name": "Catalan"}),
        ("cmn", {"name": "Mandarin Chinese"}),
        ("cym", {"name": "Welsh"}),
        ("deu", {"name": "German"}),
        ("eng", {"name": "English"}),
        ("est", {"name": "Estonian"}),
        ("fas", {"name": "Persian"}),
        ("fra", {"name": "French"}),
        ("ita", {"name": "Italian"}),
        ("kab", {"name": "Kabyle"}),
        ("kir", {"name": "Kyrgyz"}),
        ("mon", {"name": "Mongolian"}),
        ("nld", {"name": "Dutch"}),
        ("rus", {"name": "Russian"}),
        ("spa", {"name": "Spanish"}),
        ("swe", {"name": "Swedish"}),
        ("tur", {"name": "Turkish"}),
    ])

    def __init__(self, *args, **kwargs):
        #FIXME, terrible
        dataset_tree = self.__class__.dataset_tree
        if not dataset_tree:
            for label in self.default_label_definitions:
                assert label not in dataset_tree
                dataset_tree[label] = [[label, "wav"]]
        super().__init__(*args, **kwargs)
        self.file_to_speaker_id = {}
        assert "dataset_root" in kwargs, "CommonVoiceWalker always needs the dataset root for loading metadata files"
        for label in self.label_definitions:
            with open(os.path.join(kwargs["dataset_root"], label, "validated.tsv")) as f:
                for row in csv.DictReader(f, delimiter='\t'):
                    self.file_to_speaker_id[row["path"].split(".mp3")[0]] = row["client_id"]

    def parse_speaker_id(self, path):
        return self.file_to_speaker_id.get(os.path.basename(path).split(".wav")[0])


all_walkers = collections.OrderedDict({
    "ogi": OGIWalker,
    "ogi-legacy": OGIWalker2,
    "vardial-2017": VarDial2017Walker,
    "common-voice": CommonVoiceWalker,
})
