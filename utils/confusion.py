import numpy as np
import matplotlib.pyplot as plt
import itertools
from conf import *

def global_align(x, y, s_match=1, s_mismatch=-1, s_gap=-1):
    A = []
    for i in range(len(y) + 1):
        A.append([0] * (len(x) + 1))
    for i in range(len(y) + 1):
        A[i][0] = s_gap * i
    for i in range(len(x) + 1):
        A[0][i] = s_gap * i
    for i in range(1, len(y) + 1):
        for j in range(1, len(x) + 1):
            A[i][j] = max(
                A[i][j - 1] + s_gap,
                A[i - 1][j] + s_gap,
                A[i - 1][j - 1] + (s_match if (y[i - 1] == x[j - 1] and y[i - 1] != '-') else 0) + (
                    s_mismatch if (y[i - 1] != x[j - 1] and y[i - 1] != '-' and x[j - 1] != '-') else 0) + (
                    s_gap if (y[i - 1] == '-' or x[j - 1] == '-') else 0)
            )
    align_X = ""
    align_Y = ""
    i = len(x)
    j = len(y)
    while i > 0 or j > 0:
        current_score = A[j][i]
        if i > 0 and j > 0 and (
                ((x[i - 1] == y[j - 1] and y[j - 1] != '-') and current_score == A[j - 1][i - 1] + s_match) or
                ((y[j - 1] != x[i - 1] and y[j - 1] != '-' and x[i - 1] != '-') and current_score == A[j - 1][
                    i - 1] + s_mismatch) or
                ((y[j - 1] == '-' or x[i - 1] == '-') and current_score == A[j - 1][i - 1] + s_gap)
        ):
            align_X = x[i - 1] + align_X
            align_Y = y[j - 1] + align_Y
            i = i - 1
            j = j - 1
        elif i > 0 and (current_score == A[j][i - 1] + s_gap):
            align_X = x[i - 1] + align_X
            align_Y = "-" + align_Y
            i = i - 1
        else:
            align_X = "-" + align_X
            align_Y = y[j - 1] + align_Y
            j = j - 1
    return (align_X, align_Y, A[len(y)][len(x)], 1)


class ConfusionMatrix:
    def __init__(self):
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', '-']
        self.num_classes = len(self.classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def add_data(self, pred, y):
        actual, predicted, _, _ = global_align(y, pred)
        for p, a in zip(predicted, actual):
            if p in self.classes and a in self.classes:
                pred_index = self.classes.index(p)
                actual_index = self.classes.index(a)
                self.matrix[actual_index, pred_index] += 1
    def plot_matrix(self, title, path = None):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, CLASSES + ['BLANK'], rotation=45)
        plt.yticks(tick_marks, CLASSES + ['BLANK'])
        # plt.xticks(tick_marks, self.classes, rotation=45)
        # plt.yticks(tick_marks, self.classes)

        fmt = 'd'
        thresh = self.matrix.max() / 2.
        for i, j in itertools.product(range(self.matrix.shape[0]), range(self.matrix.shape[1])):
            if self.matrix[i, j] != 0:
                plt.text(j, i, format(self.matrix[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if self.matrix[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        if not path:
            plt.show()
        else:
            plt.savefig(path)

if __name__ == "__main__":
    cm = ConfusionMatrix()

    cm.add_data("ABCD", "AGCED")
    cm.add_data("ACD", "AGCD")

    cm.matrix = cm.matrix / np.sum(cm.matrix, axis = 1)
    cm.plot_matrix('test')