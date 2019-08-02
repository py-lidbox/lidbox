"""
Speech dataset walkers that can iterate over datasets.
"""
import os
import collections
import re

from speechbox.system import read_wavfile, md5sum, get_audio_type
from . import DatasetRecursionError


class SpeechDatasetWalker:
    """
    Instances of this class are iterable, yielding (label, wavpath) pairs for every file in some dataset, given the root directory of the dataset.
    The tree structure of a particular dataset is defined in the self.label_definitions dict in a subclass of this class.
    """
    def __init__(self, dataset_root=None, sampling_rate_override=None):
        # Where to start an os.walk from
        # If None, then the paths should be set with overwrite_target_paths
        self.dataset_root = dataset_root
        # If not None, an integer denoting the re-sampling rate to be applied to every wav-file
        self.sampling_rate = sampling_rate_override
        # Metadata for each label
        self.label_definitions = collections.OrderedDict()
        # Label to speaker id mapping
        self.ignored_speaker_ids_by_label = {}

    def join_root(self, *paths):
        return os.path.join(self.dataset_root, *paths)

    def overwrite_target_paths(self, paths, labels):
        """Overwrite dataset directory traversal list by absolute paths that should be walked over instead."""
        # Clear all current paths and directories
        for label_def in self.label_definitions.values():
            label_def["sample_dirs"] = []
            label_def["sample_files"] = []
        # Set all given wavpaths
        for label, path in zip(labels, paths):
            self.label_definitions[label]["sample_files"].append(path)

    def load(self, wavpath):
        return read_wavfile(wavpath, sr=self.sampling_rate)

    def make_label_to_index_dict(self):
        return {label: i for i, label in enumerate(sorted(self.label_definitions))}

    def parse_speaker_id(self, wavpath):
        raise NotImplementedError

    def count_files_per_speaker_by_label(self):
        c = {label: collections.Counter() for label in self.label_definitions}
        for label, path in iter(self):
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

    def walk(self, file_extensions=("wav",), check_duplicates=False, check_read=False, followlinks=True, verbosity=0):
        """
        Walk over all files in the dataset and yield (label, filepath) pairs.
        By default, only *.wav files will be returned.
        check_duplicates: an MD5 hash will be computed for every file, and if more than 1 files with matching hashes are found, all subsequent files with matching hashes are skipped.
        check_read: every file will be opened with librosa.core.load and skipped if that function throws an exception.
        followlinks: also walk down symbolic links.
        """
        duplicates = collections.defaultdict(list)
        num_invalid = 0
        num_walked = 0
        def audiofile_ok(wavpath, label):
            if not os.path.exists(wavpath):
                return False
            if check_read:
                audio_type = get_audio_type(wavpath)
                if verbosity and not wavpath.endswith(audio_type):
                    print("Warning: the file extension of file '{}' does not match its contents of type '{}'".format(wavpath, audio_type))
                wav, srate = read_wavfile(wavpath, sr=None)
                if verbosity and self.sampling_rate and self.sampling_rate != srate:
                    print("Warning: Dataset sampling rate override set to {} but audio file had native rate {}, file was '{}'".format(self.sampling_rate, srate, wavpath))
                if wav is None:
                    num_invalid += 1
                    if verbosity:
                        print("Warning: invalid/empty/corrupted audio file '{}', skipping to next file".format(wavpath))
                    return False
            if check_duplicates:
                content_hash = md5sum(wavpath)
                duplicates[content_hash].append(wavpath)
                if len(duplicates[content_hash]) > 1:
                    if verbosity:
                        print("Warning: Duplicate audio file '{}', its contents were already seen at least once during this walk".format(wavpath))
                        print("Audio files with shared MD5 hashes:")
                        for other_wavpath in duplicates[content_hash]:
                            print("  ", other_wavpath)
                    return False
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
            if verbosity > 1:
                if sample_dirs:
                    print("Label '{}' has {} directories that will now be fully traversed to find all samples".format(label, len(sample_dirs)))
                else:
                    print("Label '{}' has no directories containing samples".format(label))
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
                            yield label, wavpath
            # Then yield all directly specified wavpaths
            sample_files = self.label_definitions[label].get("sample_files", [])
            if verbosity > 1:
                if sample_files:
                    print("Label '{}' has {} samples defined as filepaths".format(label, len(sample_files)))
                else:
                    print("Label '{}' has no samples defined as filepaths".format(label))
            for wavpath in sample_files:
                num_walked += 1
                if audiofile_ok(wavpath, label):
                    yield label, wavpath
        if verbosity > 1:
            print("Walk finished by walker:", str(self))
        if verbosity:
            print("  Found {} audio files in total".format(num_walked))
            if check_duplicates:
                num_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
                print("  Found {} duplicate audio files".format(num_duplicates))
            if check_read:
                print("  Found {} invalid/empty/corrupted audio files".format(num_invalid))

    def parse_directory_tree(self):
        for label, definition in self.label_definitions.items():
            definition["sample_dirs"] = [self.join_root(*paths) for paths in self.dataset_tree[label]]

    def __iter__(self):
        return self.walk()

    def __repr__(self):
        return ("<{}: root={}, num_labels={}>"
                .format(self.__class__.__name__, self.dataset_root, len(self.label_definitions.keys())))

    def __str__(self):
        return ("{} starting at directory '{}', walking over data containing {} different labels"
                .format(self.__class__.__name__, self.dataset_root, len(self.label_definitions.keys())))


class OGIWalker(SpeechDatasetWalker):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_definitions = collections.OrderedDict({
            "eng": {"name": "English"},
            "hin": {"name": "Hindi"},
            "jpn": {"name": "Japanese"},
            "kor": {"name": "Korean"},
            "cmn": {"name": "Mandarin Chinese"},
            "spa": {"name": "Spanish"},
            "tam": {"name": "Tamil"},
            "vie": {"name": "Vietnamese"},
        })
        if "dataset_root" in kwargs:
            self.parse_directory_tree()
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

    def parse_speaker_id(self, path):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_definitions = collections.OrderedDict({
            "EGY": {
                "name": "Egyptian Arabic",
                "sample_dirs": [
                    self.join_root("wav", "EGY"),
                ]
            },
            "GLF": {
                "name": "Gulf Arabic",
                "sample_dirs": [
                    self.join_root("wav", "GLF"),
                ]
            },
            "LAV": {
                "name": "Levantine Arabic",
                "sample_dirs": [
                    self.join_root("wav", "LAV"),
                ]
            },
            "MSA": {
                "name": "Modern Standard Arabic",
                "sample_dirs": [
                    self.join_root("wav", "MSA"),
                ]
            },
            "NOR": {
                "name": "North African Arabic",
                "sample_dirs": [
                    self.join_root("wav", "NOR"),
                ]
            },
        })


class MGB3TestSetWalker(SpeechDatasetWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #FIXME don't do file io in the initializer
        label_to_paths = collections.defaultdict(list)
        with open(self.join_root("reference")) as wav_labels:
            for line in wav_labels:
                wavpath, label = tuple(line.strip().split())
                label_to_paths[label].append(self.join_root("wav", wavpath) + ".wav")
        self.label_definitions = collections.OrderedDict({
            "EGY": {
                "name": "Egyptian Arabic",
                "sample_dirs": [],
                "sample_files": label_to_paths["1"]
            },
            "GLF": {
                "name": "Gulf Arabic",
                "sample_dirs": [],
                "sample_files": label_to_paths["2"]
            },
            "LAV": {
                "name": "Levantine Arabic",
                "sample_dirs": [],
                "sample_files": label_to_paths["3"]
            },
            "MSA": {
                "name": "Modern Standard Arabic",
                "sample_dirs": [],
                "sample_files": label_to_paths["4"]
            },
            "NOR": {
                "name": "North African Arabic",
                "sample_dirs": [],
                "sample_files": label_to_paths["5"]
            },
        })


class TestWalker(SpeechDatasetWalker):
    dataset_tree = {
        "cmn": [["chinese"]],
        "eng": [["english"]],
        "fas": [["persian"]],
        "fra": [["french"]],
        "swe": [["swedish"]],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_definitions = collections.OrderedDict({
            "cmn": {"name": "Mandarin (China)"},
            "eng": {"name": "English"},
            "fas": {"name": "Persian"},
            "fra": {"name": "French"},
            "swe": {"name": "Swedish"},
        })
        if "dataset_root" in kwargs:
            self.parse_directory_tree()


all_walkers = collections.OrderedDict({
    "ogi": OGIWalker,
    "vardial2017": VarDial2017Walker,
    "mgb3-testset": MGB3TestSetWalker,
    "mgb3": VarDial2017Walker,
    "unittest": TestWalker,
})
