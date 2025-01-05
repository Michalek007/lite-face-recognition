from pathlib import Path
import torch

from recognition.face_recognition import FaceRecognition
from recognition.dataset import Dataset
from recognition.metrics import *


class Validation:
    def __init__(self):
        self.face_recognition = FaceRecognition(str(Path('dev\lite_face_100.pt')))
        self.dataset = Dataset.load_test(data_dir='../data_mtcnn')
        self.dataset_len = len(self.dataset)

    def validate(self, samples: int = None):
        N = self.dataset_len if not samples else samples
        assert N <= self.dataset_len
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        with torch.no_grad():
            for batch, (img1, img2, target) in enumerate(self.dataset):
                print("Current batch: ", batch)
                pred1, pred2 = self.face_recognition.model(img1), self.face_recognition.model(img2)
                tp, fp, tn, fn = get_cosine_similarity_confusion_matrix(pred1, pred2, target, 0.93)
                true_positive += tp
                false_positive += fp
                true_negative += tn
                false_negative += fn

        accuracy, recall, precision, f1_score = get_metrics(true_positive, false_positive, true_negative, false_negative)
        print(f"Accuracy: {(100*accuracy):>0.1f}%")
        print(f"Recall: {(100 * recall):>0.1f}%")
        print(f"Precision: {(100 * precision):>0.1f}%")
        print(f"F1 score: {(100 * f1_score):>0.1f}%")


if __name__ == '__main__':
    validation = Validation()
    validation.validate()
