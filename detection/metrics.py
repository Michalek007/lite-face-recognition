import torch
from torchvision.ops.boxes import box_iou
from typing import List


def get_iou_confusion_matrix(true_boxes: List[torch.Tensor], pred_boxes: List[torch.Tensor], indexes: List[int], iou_threshold: float):
    """
    Returns:
        true_positive, false_positive, true_negative, false_negative
    """
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(len(pred_boxes)):
        iou = box_iou(pred_boxes[i][..., 0:4], true_boxes[indexes[i]])
        for row_iou in iou:
            correct = (row_iou > iou_threshold).sum()
            if correct >= 1:
                true_positive += 1
            else:
                false_positive += 1
        for j in range(iou.size(1)):
            if (iou[..., j] > iou_threshold).sum() < 1:
                false_negative += 1
    indexes = set(indexes)
    for i in {j for j in range(len(true_boxes))} - indexes:
        false_negative += len(true_boxes[i])
    return true_positive, false_positive, true_negative, false_negative


def get_iou_accuracy(true_boxes: List[torch.Tensor], pred_boxes: List[torch.Tensor], indexes: List[int],  iou_threshold: float):
    true_positive, false_positive, true_negative, false_negative = get_iou_confusion_matrix(true_boxes, pred_boxes, indexes, iou_threshold)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    return accuracy


def get_iou_metrics(true_boxes: List[torch.Tensor], pred_boxes: List[torch.Tensor], indexes: List[int],  iou_threshold: float):
    """
    Returns:
        accuracy, recall, precision, f1_score
    """
    true_positive, false_positive, true_negative, false_negative = get_iou_confusion_matrix(true_boxes, pred_boxes, indexes, iou_threshold)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    recall = true_positive / (true_positive+false_negative)
    precision = true_positive / (true_positive+false_positive)
    f1_score = 2*precision*recall/(precision+recall)
    return accuracy, recall, precision, f1_score
