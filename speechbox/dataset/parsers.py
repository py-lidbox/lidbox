"""
Speech dataset parsers and cleaning tools.
"""
import collections
import csv
import os

import sox


class DatasetParser:
    """Base parser that can transform wavfiles in some directory."""
    def __init__(self, dataset_root, output_dir, output_count_limit=None, resampling_freq=None, fail_early=False):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.output_count_limit = output_count_limit
        self.resampling_freq = resampling_freq
        self.fail_early = fail_early

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
        except sox.core.SoxError:
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
        for src_path in self.iter_wavfiles_at_root():
            dst_path = os.path.join(self.output_dir, os.path.basename(src_path))
            yield src_path, self.build(t, src_path, dst_path)

    def __repr__(self):
        return "<{}: src='{}' dst='{}'>".format(self.__class__.__name__, self.dataset_root, self.output_dir)


class CommonVoiceParser(DatasetParser):
    """mp3 to wav parser for the Mozilla Common Voice dataset."""
    def __init__(self, dataset_root, *args, **kwargs):
        super().__init__(dataset_root, *args, **kwargs)
        with open(os.path.join(self.dataset_root, "validated.tsv")) as f:
            self.metadata = list(csv.DictReader(f, delimiter='\t'))

    def top_voted_samples(self, limit=None):
        def upvote_ratio(row):
            return int(row["up_votes"]) - int(row["down_votes"])
        return sorted(self.metadata, key=upvote_ratio, reverse=True)[:limit]

    def convert_to_wavs(self, samples, output_dir, resample_to=None):
        t = (sox.transform.Transformer()
                .set_globals(verbosity=2)
                .set_input_format(file_type="mp3")
                .set_output_format(file_type="wav"))
        if resample_to:
            t = t.rate(resample_to)
        for sample in samples:
            src_path = os.path.join(self.dataset_root, "clips", sample["path"])
            dst_path = os.path.join(output_dir, sample["path"].split(".mp3")[0] + ".wav")
            if os.path.exists(dst_path):
                print("Warning: Skipping '{}' because it already exists and will not be overwritten".format(dst_path))
                continue
            yield src_path, self.build(t, src_path, dst_path)

    def parse(self):
        top_samples = self.top_voted_samples(limit=self.output_count_limit)
        return self.convert_to_wavs(top_samples, self.output_dir, resample_to=self.resampling_freq)


all_parsers = collections.OrderedDict({
    "dir-of-wavs": DatasetParser,
    "common-voice": CommonVoiceParser,
})
