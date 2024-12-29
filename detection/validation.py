from pathlib import Path

from detection.lite_mtcnn import LiteMTCNN
from detection.metrics import *
from detection.dataset import Dataset


class Validation:
    def __init__(self):
        self.model = LiteMTCNN()
        self.dataset = Dataset.load_test(data_dir=str(Path('C:/Users/Public/Projects/MachineLearning/Datasets/FDDB')))
        self.dataset_len = len(self.dataset)

    def validate(self, samples: int = None):
        N = self.dataset_len if not samples else samples
        assert N <= self.dataset_len
        print(N)
        pred_boxes_list = []
        indexes = []
        true_boxes_list = []
        with torch.no_grad():
            for i in range(N):
                print(i)
                img = self.dataset[i][0]
                true_boxes = self.dataset[i][1]

                SCALE = False
                if SCALE:
                    new_width, new_height = 100, 100
                    scale_x = new_width/img.width
                    scale_y = new_height/img.height
                    img = img.resize((new_height, new_width))
                    true_boxes *= torch.tensor([scale_x, scale_y, scale_x, scale_y])

                pred_boxes = self.model.detect(img)

                # display_boxes(img, pred_boxes, true_boxes)
                if pred_boxes is not None:
                    pred_boxes_list.append(pred_boxes)
                    indexes.append(i)
                true_boxes_list.append(true_boxes)
                i += 1

            print(pred_boxes_list)
            print(indexes)
            print(true_boxes_list)

            # accuracy = get_iou_accuracy(true_boxes_list, pred_boxes_list, indexes, 0.5)
            accuracy, recall, precision, f1_score = get_iou_metrics(true_boxes_list, pred_boxes_list, indexes, 0.5)
            print(f"Accuracy: {(100*accuracy):>0.1f}%")
            print(f"Recall: {(100 * recall):>0.1f}%")
            print(f"Precision: {(100 * precision):>0.1f}%")
            print(f"F1 score: {(100 * f1_score):>0.1f}%")


if __name__ == '__main__':
    validation = Validation()
    validation.validate()
