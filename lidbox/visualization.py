import logging
import matplotlib.pyplot as plt
import numpy as np


# Modified from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def draw_confusion_matrix(cm, label_names, title='', cmap=plt.cm.Blues, no_legend=True):
    plt.style.use("default")
    num_labels = len(label_names)
    cm = np.array(cm, dtype=np.float32)
    assert cm.shape[1] == cm.shape[0] == num_labels, "Confusion matrix shape {} must match amount of labels {} both in columns and rows".format(cm.shape, num_labels)
    # Normalize by support
    cm /= cm.sum(axis=1, keepdims=True)
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
            ax.text(col, row, format(cm[row, col], '.2f'),
                    ha="center",
                    va="center",
                    color="white" if cm[row, col] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return fig, ax


def draw_spectrogram(spectra, title='', cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    im = ax.imshow(spectra.T[::-1], cmap=cmap)
    plt.tight_layout()
    return fig, ax
