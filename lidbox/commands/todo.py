class Predict(E2EBase):
    """
    Use a trained model to produce likelihoods for all target languages from all utterances in the test set.
    Writes all predictions as scores and information about the target and non-target languages into the cache dir.
    """

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("predict options")
        optional.add_argument("--score-precision", type=int, default=6)
        optional.add_argument("--score-separator", type=str, default=' ')
        optional.add_argument("--trials", type=str)
        optional.add_argument("--scores", type=str)
        optional.add_argument("--checkpoint",
            type=str,
            help="Specify which Keras checkpoint to load model weights from, instead of using the most recent one.")
        return parser

    def predict(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for prediction")
        self.model_id = self.experiment_config["experiment"]["name"]
        if not args.trials:
            args.trials = os.path.join(self.cache_dir, self.model_id, "predictions", "trials")
        if not args.scores:
            args.scores = os.path.join(self.cache_dir, self.model_id, "predictions", "scores")
        self.make_named_dir(os.path.dirname(args.trials))
        self.make_named_dir(os.path.dirname(args.scores))
        training_config = self.experiment_config["experiment"]
        feat_config = self.experiment_config["features"]
        if args.verbosity > 1:
            print("Using model parameters:")
            yaml_pprint(training_config)
            print()
        if args.verbosity > 1:
            print("Using feature extraction parameters:")
            yaml_pprint(feat_config)
            print()
        model = self.create_model(dict(training_config), skip_training=True)
        if args.verbosity > 1:
            print("Preparing model")
        labels = self.experiment_config["dataset"]["labels"]
        model.prepare(labels, training_config)
        checkpoint_dir = self.get_checkpoint_dir()
        if args.checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
        elif "best_checkpoint" in self.experiment_config.get("prediction", {}):
            checkpoint_path = os.path.join(checkpoint_dir, self.experiment_config["prediction"]["best_checkpoint"])
        else:
            checkpoints = os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []
            if not checkpoints:
                print("Error: Cannot evaluate with a model that has no checkpoints, i.e. is not trained.")
                return 1
            if "checkpoints" in training_config:
                monitor_value = training_config["checkpoints"]["monitor"]
                monitor_mode = training_config["checkpoints"].get("mode")
            else:
                monitor_value = "epoch"
                monitor_mode = None
            checkpoint_path = os.path.join(checkpoint_dir, models.get_best_checkpoint(checkpoints, key=monitor_value, mode=monitor_mode))
        if args.verbosity:
            print("Loading model weights from checkpoint file '{}'".format(checkpoint_path))
        model.load_weights(checkpoint_path)
        if args.verbosity:
            print("\nEvaluating testset with model:")
            print(str(model))
            print()
        ds = "test"
        if args.verbosity > 2:
            print("Dataset config for '{}'".format(ds))
            yaml_pprint(training_config[ds])
        ds_config = dict(training_config, **training_config[ds])
        del ds_config["train"], ds_config["validation"]
        if args.verbosity and "dataset_logger" in ds_config:
            print("Warning: dataset_logger in the test datagroup has no effect.")
        datagroup_key = ds_config.pop("datagroup")
        datagroup = self.experiment_config["dataset"]["datagroups"][datagroup_key]
        utt2path_path = os.path.join(datagroup["path"], datagroup.get("utt2path", "utt2path"))
        utt2label_path = os.path.join(datagroup["path"], datagroup.get("utt2label", "utt2label"))
        utt2path = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2path_path)
        )
        utt2label = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2label_path)
        )
        utterance_list = list(utt2path.keys())
        if args.file_limit:
            utterance_list = utterance_list[:args.file_limit]
            if args.verbosity > 3:
                print("Using utterance ids:")
                yaml_pprint(utterance_list)
        int2label = self.experiment_config["dataset"]["labels"]
        label2int, OH = tf_util.make_label2onehot(int2label)
        def label2onehot(label):
            return OH[label2int.lookup(label)]
        labels_set = set(int2label)
        paths = []
        paths_meta = []
        for utt in utterance_list:
            label = utt2label[utt]
            if label not in labels_set:
                continue
            paths.append(utt2path[utt])
            paths_meta.append((utt, label))
        if args.verbosity:
            print("Extracting test set features for prediction")
        features = self.extract_features(
            feat_config,
            "test",
        )
        conf_json, conf_checksum = config_checksum(self.experiment_config, datagroup_key)
        features = tf_data.prepare_dataset_for_training(
            features,
            ds_config,
            feat_config,
            label2onehot,
            self.model_id,
            verbosity=args.verbosity,
            conf_checksum=conf_checksum,
        )
        # drop meta wavs required only for vad
        features = features.map(lambda *t: t[:3])
        if ds_config.get("features_cache", True):
            features_cache_dir = os.path.join(self.cache_dir, "features")
        else:
            features_cache_dir = "/tmp/tensorflow-cache"
        features_cache_path = os.path.join(
            features_cache_dir,
            self.experiment_config["dataset"]["key"],
            ds,
            feat_config["type"],
            conf_checksum,
        )
        self.make_named_dir(os.path.dirname(features_cache_path), "features cache")
        if not os.path.exists(features_cache_path + ".md5sum-input"):
            with open(features_cache_path + ".md5sum-input", "w") as f:
                print(conf_json, file=f, end='')
            if args.verbosity:
                print("Writing features into new cache: '{}'".format(features_cache_path))
        else:
            if args.verbosity:
                print("Loading features from existing cache: '{}'".format(features_cache_path))
        features = features.cache(filename=features_cache_path)
        if args.verbosity:
            print("Gathering all utterance ids from features dataset iterator")
        # Gather utterance ids, this also causes the extraction pipeline to be evaluated
        utterance_ids = []
        i = 0
        if args.verbosity > 1:
            print(now_str(date=True), "- 0 samples done")
        for _, _, uttids in features.as_numpy_iterator():
            for uttid in uttids:
                utterance_ids.append(uttid.decode("utf-8"))
                i += 1
                if args.verbosity > 1 and i % 10000 == 0:
                    print(now_str(date=True), "-", i, "samples done")
        if args.verbosity > 1:
            print(now_str(date=True), "- all", i, "samples done")
        if args.verbosity:
            print("Features extracted, writing target and non-target language information for each utterance to '{}'.".format(args.trials))
        with open(args.trials, "w") as trials_f:
            for utt, target in utt2label.items():
                for lang in int2label:
                    print(lang, utt, "target" if target == lang else "nontarget", file=trials_f)
        if args.verbosity:
            print("Starting prediction with model")
        predictions = model.predict(features.map(lambda *t: t[0]))
        if args.verbosity > 1:
            print("Done predicting, model returned predictions of shape {}. Writing them to '{}'.".format(predictions.shape, args.scores))
        num_predictions = 0
        with open(args.scores, "w") as scores_f:
            print(*int2label, file=scores_f)
            for utt, pred in zip(utterance_ids, predictions):
                pred_scores = [np.format_float_positional(x, precision=args.score_precision) for x in pred]
                print(utt, *pred_scores, sep=args.score_separator, file=scores_f)
                num_predictions += 1
        if args.verbosity:
            print("Wrote {} prediction scores to '{}'.".format(num_predictions, args.scores))

    def run(self):
        super().run()
        return self.predict()



command_tree = [
    (E2E, [Predict]),
]
