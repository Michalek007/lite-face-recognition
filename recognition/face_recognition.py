import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import os
from torchvision.transforms import ToTensor, Compose, Normalize
from typing import List
from recognition.model import Model
from detection import LiteMTCNN


class FaceRecognition:
    def __init__(self, model_pt_file: str):
        self.model = Model.get('light_face_100', 3, (100, 100)).eval()
        self.model.load_state_dict(torch.load(model_pt_file, weights_only=True))
        self.lite_mtcnn = LiteMTCNN().eval()
        self.known_embeddings = []
        self.names = []
        self.transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.cosine_similarity = nn.CosineSimilarity(dim=0)
        self.margin = 0.97
        self.similarity_split = 0.5
        self.similarity_number = 4

    def add_known_person(self, files: list, name: str, is_aligned: bool = False):
        images = []
        for i, file in enumerate(files):
            image = Image.open(file)
            if is_aligned:
                images.append(image)
                continue

            aligned = self.lite_mtcnn(image)
            if aligned:
                images.append(aligned[0])

        embeddings = []

        for image in images:
            embeddings.append(self.get_embedding(image))

        self.add_known_embedding(embeddings, name)

    def add_known_embedding(self, embeddings: List[torch.Tensor], name: str):
        self.known_embeddings.append(embeddings)
        self.names.append(name)

    def recognize(self, files: list, is_aligned: bool = False):
        recognized_all = [[] for _ in range(len(files))]
        for name in self.names:
            recognized_names = self.recognize_one(files, name, is_aligned)
            for i in range(len(recognized_all)):
                recognized_all[i] += recognized_names[i]

        return recognized_all

    def recognize_embeddings(self, embeddings: list):
        recognized_all = [[] for _ in range(len(embeddings))]
        for name in self.names:
            recognized_names = self.recognize_embeddings_one(embeddings, name)
            for i in range(len(recognized_all)):
                recognized_all[i] += recognized_names[i]

        return recognized_all

    def recognize_one(self, files: list, name: str, is_aligned: bool = False):
        target_embeddings_list = self.get_target_embeddings(files, is_aligned)
        return self.recognize_embeddings_one(target_embeddings_list, name)

    def recognize_embeddings_one(self, embeddings: List[List[torch.Tensor]], name: str):
        if name not in self.names:
            raise ValueError("No person found under this name!")

        known_embedding_index = self.names.index(name)
        recognized_names = []
        with torch.no_grad():
            for j, target_embeddings in enumerate(embeddings):
                recognized_names.append([])
                for embedding in target_embeddings:
                    recognized_count = 0
                    known_face_embeddings = self.known_embeddings[known_embedding_index]
                    recognized_max_count = len(known_face_embeddings)
                    for known_embedding in known_face_embeddings:
                        if self.is_recognized(known_embedding, embedding):
                            recognized_count += 1
                    threshold = self.similarity_number if self.similarity_number else int(recognized_max_count*self.similarity_split)
                    print(recognized_count)
                    if recognized_count >= threshold:
                        recognized_names[j].append(name)
                    else:
                        recognized_names[j].append('unknown')
        return recognized_names

    def is_recognized(self, known_embedding: torch.Tensor, target_embedding: torch.Tensor):
        distance = self.cosine_similarity(known_embedding, target_embedding)
        return distance >= self.margin

    def get_embedding(self, image: Image):
        image = self.transform(image).unsqueeze(0)
        return self.model(image)[0]

    def get_target_embeddings(self, files: list, is_aligned: bool = False):
        images = []
        for file in files:
            image = Image.open(file)
            if image.width != 100 or image.height != 100:
                image = image.resize((100, 100))
            if is_aligned:
                images.append([image])
                continue
            aligned = self.lite_mtcnn(image)
            images.append(aligned)

        target_embeddings_list = []
        for aligned_images in images:
            embeddings = []
            for img in aligned_images:
                embeddings.append(self.get_embedding(img))
            target_embeddings_list.append(embeddings)
        return target_embeddings_list

    def reset_known_embeddings(self):
        self.known_embeddings = []
        self.names = []


if __name__ == '__main__':
    face_recognition = FaceRecognition('dev\\lite_face_100.pt')
    # files = ('dev\\ja.jpg', 'dev\\ja2.jpg', 'dev\\cam_admin_18.jpg', 'dev\\cam_admin_23.jpg')
    # files = ('dev\\michal.jpg', 'dev\\cam_admin_18.jpg', 'dev\\ja.jpg', 'dev\\cam_admin_39.jpg', 'dev\\cam_admin_40.jpg')

    indexes = [1, 4, 6, 8, 12]
    target_files = []
    known_files = []
    for path, _, files in os.walk('..\\data_camera_test\\michal'):
        files = list(map(lambda arg: os.path.join(path, arg), files))
        for idx in indexes:
            known_files.append(files.pop(idx))
        target_files = files

    # for path, _, files in os.walk('dev\\ja'):
    #     files = list(map(lambda arg: os.path.join(path, arg), files))
    #     known_files = files

    for path, _, files in os.walk('..\\data_camera_test\\natalia'):
        files = list(map(lambda arg: os.path.join(path, arg), files))
        # for idx in indexes:
        #     known_files.append(files.pop(idx))
        target_files += files

    # import sys; sys.exit()
    face_recognition.add_known_person(known_files, 'michal')
    # face_recognition.add_known_person(files, 'michal2')
    # print(face_recognition.known_embeddings)

    # target_path = '..\\data_camera'
    # target_path = '..\\data_camera_aligned'


    # target_files = []
    # for path, dirs, files in os.walk(target_path):
    #     target_files = list(map(lambda file: os.path.join(path, file), files))
    #     break
    # target_files = ['dev\\michal.jpg']
    N = len(target_files)

    # N = 1
    # x = torch.rand(128)
    # print(x)

    # face_recognition.similarity_number = 3
    recognized_names = face_recognition.recognize_one(target_files, 'michal', is_aligned=True)
    # recognized_names = face_recognition.recognize(target_files, is_aligned=True)
    # recognized_names = face_recognition.recognize_embeddings([[x]])
    for i in range(N):
        print()
        print(target_files[i])
        print(recognized_names[i])
