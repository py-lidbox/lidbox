import itertools

import numpy as np
import matplotlib.pyplot as plt
# Plot text as text, not curves
plt.rcParams["svg.fonttype"] = "none"


# Modified version of:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def write_confusion_matrix(cm, label_names, figpath, title='', cmap=plt.cm.Blues, no_legend=True):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title)
    if not no_legend:
        plt.colorbar(fraction=0.040, pad=0.05)
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(figpath)


def plot_mfccs(mfcc):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
