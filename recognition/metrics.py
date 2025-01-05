import torch
import torch.nn as nn


def get_cosine_similarity_confusion_matrix(predictions: torch.Tensor, predictions_2: torch.Tensor, true_values: torch.Tensor, margin: float):
    """
    Returns:
        true_positive, false_positive, true_negative, false_negative
    """
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    cosine_similarity = nn.CosineSimilarity()
    distance = cosine_similarity(predictions, predictions_2)
    for i in range(distance.size(0)):
        if distance[i] >= margin:
            if true_values[i] == 1:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if true_values[i] == 0:
                true_negative += 1
            else:
                false_negative += 1

    return true_positive, false_positive, true_negative, false_negative


def get_metrics(true_positive: int, false_positive: int, true_negative: int, false_negative: int):
    """
    Returns:
        accuracy, recall, precision, f1_score
    """
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    recall = true_positive / (true_positive+false_negative)
    precision = true_positive / (true_positive+false_positive)
    f1_score = 2*precision*recall/(precision+recall)
    return accuracy, recall, precision, f1_score
