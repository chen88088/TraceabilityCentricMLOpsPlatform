from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix

def compute_kappa(gt,predict):
    return cohen_kappa_score(gt,predict)

def compute_acc(gt,predict):
    return accuracy_score(gt,predict)

def compute_f1(gt,predict):
    return f1_score(gt,predict)

def confusion(gt, predict):
    return confusion_matrix(gt, predict)