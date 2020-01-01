import random

# import librosa.display
# import librosa.feature
import matplotlib.pyplot as plt
# Plot text as text, not curves
plt.rcParams["svg.fonttype"] = "none"
import numpy as np
# import seaborn

import speechbox.system as system


# Modified from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def draw_confusion_matrix(cm, label_names, title='', cmap=plt.cm.Blues, no_legend=True):
    num_labels = len(label_names)
    assert cm.shape[1] == cm.shape[0] == num_labels, "Confusion matrix shape {} must match amount of labels {} both in columns and rows".format(cm.shape, num_labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    if title:
        ax.set_title(title)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if not no_legend:
        ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(num_labels),
           yticks=np.arange(num_labels),
           xlim=(-0.5, num_labels - 0.5),
           ylim=(num_labels - 0.5, -0.5),
           xticklabels=label_names,
           yticklabels=label_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for row in range(num_labels):
        for col in range(num_labels):
            ax.text(col, row, format(cm[row, col], 'd'),
                    ha="center",
                    va="center",
                    color="white" if cm[row, col] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return fig, ax

#TODO
def draw_training_metrics_from_tf_events(event_data, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.set(ylabel=ylabel, xlabel=xlabel, title=title)
    for data in event_data:
        model_id, event_file, metric = data["model_id"], data["event_file"], data["metric"]
        values = [value for key, value in system.iter_log_events(event_file) if key == metric]
        epochs = list(range(len(values)))
        ax.plot(epochs, values, label=model_id)
    ax.legend()
    plt.tight_layout()
    return fig, ax

#TODO
def plot_overview(wav, figpath):
    """Plot as much visual information as possible for a wavfile."""
    plt.clf()
    signal, sample_rate = wav
    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
    Sp = [librosa.power_to_db(S**p, ref=np.max) for p in (1.0, 1.5, 2.0)]
    for i, S in enumerate(Sp, start=1):
        plt.subplot(2, len(Sp), i)
        librosa.display.specshow(S, fmax=16000)
    for i, S in enumerate(Sp, start=1):
        plt.subplot(2, len(Sp), i + len(Sp))
        librosa.display.specshow(librosa.feature.mfcc(y=signal, sr=sample_rate, S=S))
    plt.tight_layout()
    plt.savefig(figpath)


def plot_features_samples(dataset_by_label, num_samples, figpath=None):
    fig, all_axes = plt.subplots(num_samples, len(dataset_by_label), figsize=(15 * len(dataset_by_label), 5 * num_samples))
    if all_axes.ndim == 1:
        all_axes = all_axes.reshape((1, -1))
    heatmap_kwargs = {
        "center": 0,
        "xticklabels": 50,
    }
    labels = list(dataset_by_label.keys())
    for sample_idx, axes in enumerate(all_axes):
        assert len(axes) == len(labels), "unexpected subplot axes shape {}".format(all_axes.shape)
        print("plotting sample {} for {} labels".format(sample_idx, len(labels)))
        for ax, label in zip(axes, labels):
            features = dataset_by_label[label]
            assert len(features) >= num_samples, "Too few examples ({}) in dataset to draw {} samples".format(len(features), num_samples)
            sample = features[sample_idx]
            assert sample.ndim in (2, 3), "this plotter supports only samples with 2 dimensions"
            # flatten 1-dim channels
            if sample.ndim == 3:
                assert sample.shape[2] == 1, "heatmap implemented only for 2-dim features (no reduction of channels in 3-dims)"
                sample = sample.reshape((sample.shape[0], sample.shape[1]))
            seaborn.heatmap(sample.T, ax=ax, **heatmap_kwargs)
            ax.set_title("{} sample {}".format(label, sample_idx))
            ax.invert_yaxis()
    plt.tight_layout()
    if figpath is None:
        plt.show()
    else:
        plt.savefig(figpath)
        print("Wrote figure to '{}'".format(figpath))

def plot_sequence_features_sample(dataset_by_label, figpath=None, sample_width=None):
    if sample_width is None:
        sample_width = 32
    fig, axes = plt.subplots(2, int(np.ceil(len(dataset_by_label)/2)), figsize=(20, 15))
    heatmap_kwargs = {
        "center": 0,
        "xticklabels": 50,
    }
    for ax, (label, features) in zip(axes.reshape(-1), dataset_by_label.items()):
        assert features.ndim > 2, "Cannot plot single-dim features as sequences"
        assert len(features) - sample_width > 0, "Too few sequences, cannot draw sample"
        rand_indexes = np.random.choice(np.arange(features.shape[0]), size=sample_width, replace=False)
        sample = features[rand_indexes].reshape((-1, features.shape[-1] if features.shape[-1] > 1 else features.shape[-2]))
        seaborn.heatmap(sample.T, ax=ax, **heatmap_kwargs)
        ax.set_title(label)
        ax.invert_yaxis()
    plt.tight_layout()
    if figpath is None:
        plt.show()
    else:
        plt.savefig(figpath)
        print("Wrote figure to '{}'".format(figpath))
