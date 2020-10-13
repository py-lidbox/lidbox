"""
Extract embeddings using a trained model.
Usage:
    python3 get_embeddings.py config.xvector-NB.yaml
"""
import sys
import lidbox.api

def main(config_path):
    split2meta, labels, config = lidbox.api.load_splits_from_config_file(config_path)
    split2ds = lidbox.api.create_datasets(split2meta, labels, config)
    split2numpy_ds, target2label = lidbox.api.extract_embeddings_as_numpy_data(split2ds, labels)
    train = split2numpy_ds["train"]
    print(train["X"].shape)
    print(train["y"].shape)
    print(train["ids"].shape)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "first argument should be config yaml path"
    main(sys.argv[1])
