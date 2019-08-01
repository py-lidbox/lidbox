"""
Speech dataset walkers that can iterate over datasets.
"""
import os
import collections
import re

from speechbox.system import read_wavfile
from . import DatasetRecursionError


class SpeechDatasetWalker:
    """
    Instances of this class are iterable, yielding (language, wavpath) pairs for every file in some dataset, given the root directory of the dataset.
    The tree structure of a particular dataset is defined in the self.language_definitions dict in a subclass of this class.
    """
    def __init__(self, dataset_root, sampling_rate_override=None):
        # Where to start an os.walk from
        self.dataset_root = dataset_root
        # If not None, an integer denoting the re-sampling rate to be applied to every wav-file
        self.sampling_rate = sampling_rate_override
        # Metadata for each language
        self.language_definitions = collections.OrderedDict()
        # Language label to speaker id mapping
        self.label_to_ids = {}

    def join_root(self, *paths):
        return os.path.join(self.dataset_root, *paths)

    def overwrite_target_paths(self, paths, labels):
        """Overwrite dataset directory traversal list by absolute paths that should be walked over instead."""
        # Clear all current paths and directories
        for lang_def in self.language_definitions.values():
            lang_def["langdirs"] = []
            lang_def["langfiles"] = []
        # Set all given wavpaths
        for label, path in zip(labels, paths):
            self.language_definitions[label]["langfiles"].append(path)

    def load(self, wavpath):
        return read_wavfile(wavpath, sr=self.sampling_rate)

    def make_label_to_index_dict(self):
        return {label: i for i, label in enumerate(sorted(self.language_definitions))}

    def parse_speaker_id(self, wavpath):
        raise NotImplementedError

    def speaker_id_is_ignored(self, wavpath, label):
        if label not in self.label_to_ids:
            return False
        return self.parse_speaker_id(wavpath) not in self.label_to_ids[label]

    def label_to_bcp47(self, label):
        """Language mapping to BCP-47 identifiers.
        Specification: https://tools.ietf.org/html/bcp47
        """
        raise NotImplementedError

    def walk(self, file_extensions=("wav",), check_duplicates=False, check_read=False, followlinks=True, verbose=False):
        """
        Walk over all files in the dataset and yield (language key, file path) pairs.
        By default, only *.wav files will be returned.
        check_duplicates: an MD5 hash will be computed for every file, and if more than 1 files with matching hashes are found, all subsequent files with matching hashes are skipped.
        check_read: every file will be opened with librosa.core.load and skipped if that function throws an exception.
        followlinks: also walk down symbolic links.
        """
        if verbose:
            print("Starting walk with walker:", str(self))
            print("  Yielding audio files with extensions:", ', '.join(file_extensions))
            print("  Checking for duplicate audio files by hash:", check_duplicates)
            print("  Checking for invalid audio files by opening them:", check_read)
            if check_read and self.sampling_rate:
                print("  Checking that all files have sampling rate:", self.sampling_rate)
        duplicates = collections.defaultdict(list)
        num_invalid = 0
        def file_ok_can_yield(wavpath, lang):
            if self.speaker_id_is_ignored(wavpath, lang):
                return False
            if not any(wavpath.endswith(ext) for ext in file_extensions):
                return False
            if check_read:
                wav, srate = read_wavfile(wavpath, sr=None)
                if self.sampling_rate and self.sampling_rate != srate:
                    print("Warning: Dataset sampling rate override set to {} but audio file had native rate {}, file was '{}'".format(self.sampling_rate, srate, wavpath))
                if wav is None:
                    num_invalid += 1
                    print("Warning: invalid/empty/corrupted audio file '{}', skipping to next file".format(wavpath))
                    return False
            if check_duplicates:
                content_hash = md5sum(wavpath)
                duplicates[content_hash].append(wavpath)
                if len(duplicates[content_hash]) > 1:
                    print("Warning: Duplicate audio file '{}', its contents were already seen at least once during this walk".format(wavpath))
                    print("Audio files with shared MD5 hashes:")
                    for other_wavpath in duplicates[content_hash]:
                        print("  ", other_wavpath)
                    return False
            return True
        for lang in sorted(self.language_definitions.keys()):
            # First walk over all files in all directories specified to contain audio files corresponding to label 'lang'
            langdirs = self.language_definitions[lang].get("langdirs", [])
            if verbose:
                print("Label '{}' has {} directories that will now be walked over".format(lang, len(langdirs)))
            for langdir in langdirs:
                seen_directories = set()
                for parent, _, files in os.walk(langdir, followlinks=followlinks):
                    this_dir = self.join_root(parent)
                    if this_dir in seen_directories:
                        raise DatasetRecursionError(this_dir + " already traversed at least once. Does the dataset directory contain symbolic links pointing to parent directories?")
                    seen_directories.add(this_dir)
                    for f in files:
                        wavpath = self.join_root(parent, f)
                        if file_ok_can_yield(wavpath, lang):
                            yield lang, wavpath
            # Then yield all directly specified wavpaths
            langfiles = self.language_definitions[lang].get("langfiles", [])
            if verbose:
                print("Label '{}' has {} files that will now be yielded".format(lang, len(langfiles)))
            for wavpath in langfiles:
                if file_ok_can_yield(wavpath, lang):
                    yield lang, wavpath
        if verbose:
            print("Walk finished by walker:", str(self))
            if check_duplicates:
                num_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
                print("  Found {} duplicate audio files".format(num_duplicates))
            if check_read:
                print("  Found {} invalid/empty/corrupted audio files".format(num_invalid))

    def __iter__(self):
        return self.walk()

    def __repr__(self):
        return ("<{}: root={}, num_labels={}>"
                .format(self.__class__.__name__, self.dataset_root, len(self.language_definitions.keys())))

    def __str__(self):
        return ("{} starting at directory '{}', walking over data containing {} different language labels"
                .format(self.__class__.__name__, self.dataset_root, len(self.language_definitions.keys())))


class OGIWalker(SpeechDatasetWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_definitions = collections.OrderedDict({
            "eng": {
                "name": "English",
                "langdirs": [
                    self.join_root("cd01", "speech", "english"),
                ],
            },
            "hin": {
                "name": "Hindi",
                "langdirs": [
                    self.join_root("cd02", "speech", "hindi"),
                    self.join_root("cd03", "speech", "hindi")
                ],
            },
            "jpn": {
                "name": "Japanese",
                "langdirs": [
                    self.join_root("cd02", "speech", "japanese"),
                    self.join_root("cd03", "speech", "japanese")
                ],
            },
            "kor": {
                "name": "Korean",
                "langdirs": [
                    self.join_root("cd02", "speech", "korean"),
                    self.join_root("cd03", "speech", "korean")
                ],
            },
            "cmn": {
                "name": "Mandarin Chinese",
                "langdirs": [
                    self.join_root("cd02", "speech", "mandarin"),
                    self.join_root("cd03", "speech", "mandarin")
                ],
            },
            "spa": {
                "name": "Spanish",
                "langdirs": [
                    self.join_root("cd02", "speech", "spanish"),
                    self.join_root("cd03", "speech", "spanish"),
                    self.join_root("cd04", "speech", "spanish")
                ],
            },
            "tam": {
                "name": "Tamil",
                "langdirs": [
                    self.join_root("cd04", "speech", "tamil")
                ],
            },
            "vie": {
                "name": "Vietnamese",
                "langdirs": [
                    self.join_root("cd04", "speech", "vietnamese")
                ],
            },
        })
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

    def count_files_per_speaker_by_label(self):
        c = {lang: collections.Counter() for lang in self.language_definitions}
        for lang, path in iter(self):
            speaker_id = self.parse_speaker_id(path)
            c[lang][speaker_id] += 1
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
        self.label_to_ids = {label: set(ids) for label, ids in speaker_ids_by_label.items()}

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

    def label_to_bcp47(self, label):
        return self.bcp47_mappings[label]


class VarDial2017Walker(SpeechDatasetWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_definitions = collections.OrderedDict({
            "EGY": {
                "name": "Egyptian Arabic",
                "langdirs": [
                    self.join_root("wav", "EGY"),
                ]
            },
            "GLF": {
                "name": "Gulf Arabic",
                "langdirs": [
                    self.join_root("wav", "GLF"),
                ]
            },
            "LAV": {
                "name": "Levantine Arabic",
                "langdirs": [
                    self.join_root("wav", "LAV"),
                ]
            },
            "MSA": {
                "name": "Modern Standard Arabic",
                "langdirs": [
                    self.join_root("wav", "MSA"),
                ]
            },
            "NOR": {
                "name": "North African Arabic",
                "langdirs": [
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
        self.language_definitions = collections.OrderedDict({
            "EGY": {
                "name": "Egyptian Arabic",
                "langdirs": [],
                "langfiles": label_to_paths["1"]
            },
            "GLF": {
                "name": "Gulf Arabic",
                "langdirs": [],
                "langfiles": label_to_paths["2"]
            },
            "LAV": {
                "name": "Levantine Arabic",
                "langdirs": [],
                "langfiles": label_to_paths["3"]
            },
            "MSA": {
                "name": "Modern Standard Arabic",
                "langdirs": [],
                "langfiles": label_to_paths["4"]
            },
            "NOR": {
                "name": "North African Arabic",
                "langdirs": [],
                "langfiles": label_to_paths["5"]
            },
        })


all_walkers = collections.OrderedDict({
    "ogi": OGIWalker,
    "vardial2017": VarDial2017Walker,
    "mgb3-testset": MGB3TestSetWalker,
    "mgb3": VarDial2017Walker,
})
