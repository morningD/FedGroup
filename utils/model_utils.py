from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from collections import OrderedDict

def fscore(y_true, y_pred):
    # For ABIDE, Austism Label = 1-1 = 0
    return f1_score(y_true, y_pred, pos_label=0)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def adjusted_balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred, adjusted=True)

def calculate_model_state_difference(mst0, mst1):
    return OrderedDict({k: v1-v0 for k, v0, v1 in zip(mst0.keys(), mst0.values(), mst1.values())})
