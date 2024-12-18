from model import Model
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
import os
from torchvision.transforms import ToTensor, Compose, Normalize
from detection import LiteMTCNN


class FaceRecognition:
    def __init__(self, model_pt_file: str):
        self.model = Model.get('light_face_100', 3, (100, 100)).eval()
        self.model.load_state_dict(torch.load(model_pt_file, weights_only=True))
        # self.mtcnn = MTCNN(keep_all=True, image_size=100)
        self.lite_mtcnn = LiteMTCNN().eval()
        self.known_faces = []
        self.names = []
        self.transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.cosine_similarity = nn.CosineSimilarity(dim=0)
        self.margin = 0.95

    def add_known_person(self, files: list, name: str, is_aligned: bool = False):
        temp_path = Path('temp')
        temp_path.mkdir(exist_ok=True)
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
            image = self.transform(image).unsqueeze(0)
            embeddings.append(self.model(image)[0])

        self.known_faces.append(embeddings)
        self.names.append(name)

    def recognize(self, files: list, is_aligned: bool = False):
        images = []
        for file in files:
            image = Image.open(file)
            if is_aligned:
                images.append(image)
                continue
            aligned = self.lite_mtcnn(image)
            images.append(aligned)

        target_embeddings_list = []
        for aligned_images in images:
            embeddings = []
            for image in aligned_images:
                image = self.transform(image).unsqueeze(0)
                embeddings.append(self.model(image)[0])
            target_embeddings_list.append(embeddings)

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

    def is_recognized(self, known_embedding: torch.Tensor, target_embedding: torch.Tensor):
        return self.cosine_similarity(known_embedding, target_embedding) >= self.margin


if __name__ == '__main__':
    face_recognition = FaceRecognition('dev\\lite_face_100.pt')
    files = ('dev\\ja.jpg', 'dev\\ja2.jpg', 'dev\\cam_admin_18.jpg', 'dev\\cam_admin_23.jpg')
    face_recognition.add_known_person(files, 'michal')
    print(face_recognition.known_faces)

    target_path = '..\\data_camera'

    target_files = []
    for path, dirs, files in os.walk(target_path):
        target_files = list(map(lambda file: os.path.join(path, file), files))
        break

    recognized_names = face_recognition.recognize(target_files)
    # print(target_files)
    # print(recognized_names)
    for i in range(len(files)):
        print()
        print(target_files[i])
        print(recognized_names[i])
