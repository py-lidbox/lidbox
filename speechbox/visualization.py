import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
# Plot text as text, not curves
plt.rcParams["svg.fonttype"] = "none"
import numpy as np


# Modified from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def draw_confusion_matrix(cm, label_names, title='', cmap=plt.cm.Blues, no_legend=True):
    num_labels = len(label_names)
    assert cm.shape[0] == num_labels and cm.shape[1] == num_labels, "Invalid confusion matrix and/or labels"
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
