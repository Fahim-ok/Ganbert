from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(y_true, y_pred, labels=None):
    """
    Evaluates classification metrics.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of class labels.
    Returns:
        A dictionary with accuracy, precision, recall, and F1-score.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels, figsize=(8, 6), cmap='Blues'):
    """
    Plots the confusion matrix.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of class labels.
        figsize: Tuple for figure size.
        cmap: Color map for the heatmap.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
