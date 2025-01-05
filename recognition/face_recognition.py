import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
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
        # self.mtcnn = MTCNN(keep_all=True, image_size=100, post_process=False)
        self.lite_mtcnn = LiteMTCNN().eval()
        self.known_faces = []
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
        self.known_faces.append(embeddings)
        self.names.append(name)

    def recognize(self, files: list, is_aligned: bool = False):
        target_embeddings_list = self.get_target_embeddings(files, is_aligned)
        recognized_names = []
        with torch.no_grad():
            for j, target_embeddings in enumerate(target_embeddings_list):
                recognized_names.append([])
                for embedding in target_embeddings:
                    # print()
                    for i, known_face_embeddings in enumerate(self.known_faces):
                        for known_embedding in known_face_embeddings:
                            if self.is_recognized(known_embedding, embedding):
                                # print(f"{self.names[i]} found in image: {files[j]}")
                                recognized_names[j].append(self.names[i])
                            else:
                                # print(f"Unknown person found in image: {files[j]}")
                                recognized_names[j].append('unknown')
                        if not recognized_names[j]:
                            recognized_names[j].append("no face detected")
        return recognized_names

    def recognize_one(self, files: list, name: str, is_aligned: bool = False):
        if name not in self.names:
            raise ValueError("No person found under this name!")

        known_face_index = self.names.index(name)
        target_embeddings_list = self.get_target_embeddings(files, is_aligned)
        recognized_names = []
        with torch.no_grad():
            for j, target_embeddings in enumerate(target_embeddings_list):
                recognized_names.append([])
                for embedding in target_embeddings:
                    recognized_count = 0
                    known_face_embeddings = self.known_faces[known_face_index]
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


if __name__ == '__main__':
    face_recognition = FaceRecognition('dev\\lite_face_100.pt')
    # files = ('dev\\ja.jpg', 'dev\\ja2.jpg', 'dev\\cam_admin_18.jpg', 'dev\\cam_admin_23.jpg')
    files = ('dev\\michal.jpg', 'dev\\cam_admin_18.jpg', 'dev\\ja.jpg', 'dev\\cam_admin_39.jpg', 'dev\\cam_admin_40.jpg')
    face_recognition.add_known_person(files, 'michal')
    # print(face_recognition.known_faces)

    # target_path = '..\\data_camera'
    target_path = '..\\data_camera_aligned'

    target_files = []
    for path, dirs, files in os.walk(target_path):
        target_files = list(map(lambda file: os.path.join(path, file), files))
        break

    # recognized_names = face_recognition.recognize(target_files)
    face_recognition.similarity_number = 3
    recognized_names_one = face_recognition.recognize_one(target_files, 'michal', is_aligned=True)
    # print(target_files)
    # print(recognized_names)
    for i in range(len(files)):
        print()
        print(target_files[i])
        # print(recognized_names[i])
        print(recognized_names_one[i])
