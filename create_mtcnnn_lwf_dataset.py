from facenet_pytorch import MTCNN
from pathlib import Path
from PIL import Image
import os


# mtcnn = MTCNN(image_size=100)

# basedir = Path('C:/Users/Public/Projects/MachineLearning/lite-face-recognition/data/lfw-py/lfw-deepfunneled/')
# for path, dirs, files in os.walk(basedir):
#     for file in files:
#         if '.jpg' in file:
#             print(file)
#             img_path = os.path.join(path, file)
#             mtcnn_img_path = Path(os.path.join(path.replace('data', 'data_mtcnn'), file))
#             img = Image.open(img_path)
#             if not mtcnn_img_path.exists():
#                 mtcnn_img_path.mkdir()
#             mtcnn(img, save_path=str(mtcnn_img_path))


# img_path = Path('C:/Users/Public/Projects/MachineLearning/lite-face-recognition/data/lfw-py/lfw-deepfunneled/Marilyn_Monroe/Marilyn_Monroe_0001.jpg')
# mtcnn_img_path = Path('C:/Users/Public/Projects/MachineLearning/lite-face-recognition/data_mtcnn/lfw-py/lfw-deepfunneled/Marilyn_Monroe/Marilyn_Monroe_0001.jpg')
#
# if mtcnn_img_path.exists():
#     mtcnn_img_path.mkdir()
#
#
# img = Image.open(img_path)
# boxes = mtcnn(img, save_path=str(mtcnn_img_path))
# print(boxes)


basedirs = (
    Path('C:/Users/Public/Projects/MachineLearning/lite-face-recognition/data/lfw-py/lfw-deepfunneled/'),
    Path('C:/Users/Public/Projects/MachineLearning/lite-face-recognition/data_mtcnn/lfw-py/lfw-deepfunneled/')
)

for basedir in basedirs:
    count = 0
    for path, dirs, files in os.walk(basedir):
        count += len(files)
    print(basedir, count)

