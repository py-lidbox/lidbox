"""
Speech dataset parsers and cleaning tools.
"""
import collections
import csv
import os
import sys

# import sox


class DatasetParser:
    """Base parser that can transform wavfiles in some directory."""
    def __init__(self, dataset_root, output_dir, verbosity=0, output_count_limit=None, output_duration_limit=None, resampling_freq=None, fail_early=False, min_duration_ms=None, normalize_volume=None):
        self.dataset_root = dataset_root
        self.verbosity = verbosity
        self.output_dir = output_dir
        self.output_count_limit = output_count_limit
        self.output_duration_limit = output_duration_limit
        self.resampling_freq = resampling_freq
        self.fail_early = fail_early
        self.min_duration = min_duration_ms
        if self.min_duration is not None:
            self.min_duration *= 1e-3
        self.normalize_volume = normalize_volume

    def iter_wavfiles_at_root(self):
        """Yield all wavfiles at self.dataset_root."""
        with os.scandir(self.dataset_root) as entries:
            for entry in entries:
                if not entry.name.startswith('.') and entry.is_file():
                    wavpath = os.path.join(self.dataset_root, entry.name)
                    if sox.file_info.file_type(wavpath) == "wav":
                        yield wavpath

    def build(self, transformer, src_path, dst_path):
        output = None
        try:
            output = transformer.build(src_path, dst_path, return_output=True)
        except (sox.core.SoxError, OSError):
            if self.fail_early:
                raise
        return output

    def parse(self):
        t = (sox.transform.Transformer()
                .set_globals(verbosity=2)
                .set_input_format(file_type="wav")
                .set_output_format(file_type="wav"))
        if self.resampling_freq:
            t = t.rate(self.resampling_freq)
        if self.normalize_volume is not None:
            t = t.norm(db_level=self.normalize_volume)
        for src_path in self.iter_wavfiles_at_root():
            dst_path = os.path.join(self.output_dir, os.path.basename(src_path))
            yield src_path, self.build(t, src_path, dst_path)

    def __repr__(self):
        return "{}(dataset_root='{}', output_dir='{}')".format(self.__class__.__name__, self.dataset_root, self.output_dir)


class CommonVoiceParser(DatasetParser):
    """mp3 to wav parser for the Mozilla Common Voice dataset."""
    def __init__(self, dataset_root, *args, **kwargs):
        super().__init__(dataset_root, *args, **kwargs)
        with open(os.path.join(self.dataset_root, "validated.tsv")) as f:
            self.metadata = list(csv.DictReader(f, delimiter='\t'))

    def top_voted_samples(self):
        def upvote_ratio(row):
            return int(row["up_votes"]) - int(row["down_votes"])
        return sorted(self.metadata, key=upvote_ratio, reverse=True)

    def convert_to_wavs(self, samples, output_dir):
        t = (sox.transform.Transformer()
                .set_globals(verbosity=2)
                .set_input_format(file_type="mp3")
                .set_output_format(file_type="wav"))
        if self.resampling_freq:
            t = t.rate(self.resampling_freq)
        if self.normalize_volume is not None:
            t = t.norm(db_level=self.normalize_volume)
        total_duration = 0.0
        for sample in samples:
            src_path = os.path.join(self.dataset_root, "clips", sample["path"])
            try:
                duration = sox.file_info.duration(src_path)
            except sox.core.SoxiError:
                print("Skipping '{}' due to SoXI error".format(src_path))
                continue
            if self.min_duration is not None and duration < self.min_duration:
                if self.verbosity > 2:
                    print("Skipping '{}' because it is too short".format(src_path), file=sys.stderr)
                continue
            if self.output_duration_limit is not None and total_duration >= self.output_duration_limit:
                if self.verbosity > 1:
                    print("Stopping parse, {:.3f} second limit reached at {:.3f} seconds".format(self.output_duration_limit, total_duration))
                break
            total_duration += duration
            dst_path = os.path.join(output_dir, sample["path"].split(".mp3")[0] + ".wav")
            if os.path.exists(dst_path):
                if self.verbosity:
                    print("Warning: Skipping '{}' because it already exists".format(dst_path), file=sys.stderr)
                continue
            yield src_path, self.build(t, src_path, dst_path)

    def parse(self):
        converter = self.convert_to_wavs(self.top_voted_samples(), self.output_dir)
        num_parsed = 0
        for parsed in converter:
            if all(p is not None for p in parsed):
                yield parsed
                num_parsed += 1
            if self.output_count_limit and num_parsed >= self.output_count_limit:
                return


all_parsers = collections.OrderedDict({
    "dir-of-wavs": DatasetParser,
    "common-voice": CommonVoiceParser,
})
