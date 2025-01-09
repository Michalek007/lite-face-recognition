from pathlib import Path
import torch
import os
from PIL import Image

from recognition.face_recognition import FaceRecognition
from recognition.dataset import Dataset
from recognition.metrics import *


class Validation:
    def __init__(self):
        self.face_recognition = FaceRecognition(str(Path('dev/lite_face_100.pt')))
        self.dataset = Dataset.load_test(data_dir='../data_mtcnn')
        self.dataset_len = len(self.dataset)

    def get_metrics(self, true_positive: int, false_positive: int, true_negative: int, false_negative: int, print_metrics: bool = True, print_error_matrix: bool = True):
        if print_error_matrix:
            print("True positives: ", true_positive)
            print("False positives: ", false_positive)
            print("True negatives: ", true_negative)
            print("False negatives: ", false_negative)

        accuracy, recall, precision, f1_score = get_metrics(true_positive, false_positive, true_negative, false_negative)

        if print_metrics:
            print(f"Accuracy: {(100*accuracy):>0.1f}%")
            print(f"Recall: {(100 * recall):>0.1f}%")
            print(f"Precision: {(100 * precision):>0.1f}%")
            print(f"F1 score: {(100 * f1_score):>0.1f}%")
        return accuracy, recall, precision, f1_score

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
                tp, fp, tn, fn = get_cosine_similarity_confusion_matrix(pred1, pred2, target, 0.94)
                true_positive += tp
                false_positive += fp
                true_negative += tn
                false_negative += fn

        return self.get_metrics(true_positive, false_positive, true_negative, false_negative)

    def validate_camera_dataset(self):
        indexes = [
            [1, 4, 6, 8, 12],
            [9, 14, 0, 3, 13]
        ]
        known_files = []
        all_files = []
        dirs = ['michal', 'natalia', 'michal_natalia']
        for i, dir_name in enumerate(dirs):
            for path, _, files in os.walk(f'..\\data_camera_test\\{dir_name}'):
                files = list(map(lambda arg: os.path.join(path, arg), files))
                if i < 2:
                    known_files.append([files.pop(idx) for idx in indexes[i]])
                all_files.append(files)

        # print(all_files)
        print(known_files)
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        self.face_recognition.similarity_number = 4
        self.face_recognition.margin = 0.97
        names = ('michal', 'natalia')
        for n, name in enumerate(names):
            self.face_recognition.add_known_person(known_files[n], name)
            for k in range(len(all_files)):

                recognized_names = self.face_recognition.recognize_one(all_files[k], name)
                print(all_files[k])
                print(recognized_names)
                for i in range(len(recognized_names)):
                    if dirs[k] == name:
                        if recognized_names[i][0] == name:
                            true_positive += 1
                        else:
                            false_negative += 1
                    else:
                        if len(dirs[k].split('_')) == 2:
                            michal, natalia = dirs[k].split('_')
                            target = michal if name == michal else natalia
                            index = 0 if name == michal else 1
                            second_index = 1 if name == michal else 0
                            if recognized_names[i][index] == target:
                                true_positive += 1
                            else:
                                false_negative += 1
                            if recognized_names[i][second_index] == 'unknown':
                                true_negative += 1
                            else:
                                false_positive += 1
                        else:
                            if recognized_names[i][0] == 'unknown':
                                true_negative += 1
                            else:
                                false_positive += 1

            break

        return self.get_metrics(true_positive, false_positive, true_negative, false_negative)

    def validate_system(self):
        known_files = []
        all_files = []
        dirs = ['michal_db', 'natalia_db']
        target_dirs = ['michal', 'natalia', 'michal_natalia']
        for i, dir_name in enumerate(dirs):
            for path, _, files in os.walk(f'..\\data_camera_system_test\\{dir_name}'):
                files = list(map(lambda arg: os.path.join(path, arg), files))
                known_files.append(files)

        for i, dir_name in enumerate(target_dirs):
            for path, _, files in os.walk(f'..\\data_camera_system_test\\{dir_name}'):
                files = list(map(lambda arg: os.path.join(path, arg), files))
                all_files.append(files)

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        self.face_recognition.similarity_number = 4
        self.face_recognition.margin = 0.97
        names = ('michal', 'natalia')
        for n, name in enumerate(names):
            self.face_recognition.add_known_person(known_files[n], name)
            for k in range(len(all_files)):

                recognized_names = self.face_recognition.recognize_one(all_files[k], name)
                print(all_files[k])
                print(recognized_names)
                for i in range(len(recognized_names)):
                    if target_dirs[k] == name:
                        if recognized_names[i][0] == name:
                            true_positive += 1
                        else:
                            false_negative += 1
                    else:
                        if len(target_dirs[k].split('_')) == 2:
                            michal, natalia = target_dirs[k].split('_')
                            target = michal if name == michal else natalia
                            index = 0 if name == michal else 1
                            second_index = 1 if name == michal else 0
                            if recognized_names[i][index] == target:
                                true_positive += 1
                            else:
                                false_negative += 1
                            if recognized_names[i][second_index] == 'unknown':
                                true_negative += 1
                            else:
                                false_positive += 1
                        else:
                            if recognized_names[i][0] == 'unknown':
                                true_negative += 1
                            else:
                                false_positive += 1

        return self.get_metrics(true_positive, false_positive, true_negative, false_negative)


if __name__ == '__main__':
    validation = Validation()
    # validation.validate()
    validation.validate_camera_dataset()
    validation.validate_system()
