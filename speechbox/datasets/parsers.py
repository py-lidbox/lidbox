"""
Speech dataset parsers and cleaning tools.
"""
import collections
import csv
import os
import logging

import sox

# silence all sox logging
sox_logger = logging.getLogger("sox")
for handler in sox_logger.handlers:
    sox_logger.removeHandler(handler)


class DatasetParser:
    def __init__(self, dataset_root, output_dir, output_count_limit=None, resampling_rate=None):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.output_count_limit = output_count_limit
        self.resampling_rate = resampling_rate

    def parse(self):
        pass

    def __repr__(self):
        return "<{}: src='{}' dst='{}'>".format(self.__class__.__name__, self.dataset_root, self.output_dir)


class CommonVoiceParser(DatasetParser):
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
            yield t.build(src_path, dst_path, return_output=True)

    def parse(self):
        top_samples = self.top_voted_samples(limit=self.output_count_limit)
        return self.convert_to_wavs(top_samples, self.output_dir, resample_to=self.resampling_rate)


all_parsers = collections.OrderedDict({
    "common-voice": CommonVoiceParser,
})
