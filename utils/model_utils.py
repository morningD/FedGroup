from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

def fscore(y_true, y_pred):
    # For ABIDE, Austism Label = 1-1 = 0
    return f1_score(y_true, y_pred, pos_label=0)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def adjusted_balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred, adjusted=True)
