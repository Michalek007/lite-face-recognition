import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.transforms import ToTensor
import numpy as np


class FDDBDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str = 'train', transform=None, target_transform=None, scale: tuple = None):
        self.basedir = Path(dataset_dir)
        self.images_dir = Path.joinpath(self.basedir, 'originalPics')
        self.labels_dir = Path.joinpath(self.basedir, 'FDDB-folds')

        self.transform = transform
        self.target_transform = target_transform

        self.scale = scale

        self.images = []
        self.bboxes = []
        self._set_images()

        dataset_size = len(self)
        split_index = int(0.8*dataset_size)
        if split == 'train':
            self.images, self.bboxes = self.images[:split_index], self.bboxes[:split_index]
        elif split == 'test':
            self.images, self.bboxes = self.images[split_index:], self.bboxes[split_index:]
        else:
            pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = Path.joinpath(self.images_dir, self.images[idx])
        image = self._read_image(img_path)
        boxes = self.bboxes[idx]

        if not isinstance(self.transform, ToTensor):
            self.scale = None
        img_h = image.height
        img_w = image.width

        if self.transform:
            image = self.transform(image)

            if self.scale:
                image = interpolate(image.unsqueeze(0), self.scale).squeeze()
                scale_y = self.scale[0] / img_h
                scale_x = self.scale[1] / img_w
                scaled_boxes = []
                for x, y, x2, y2 in boxes:
                    scaled_boxes.append(
                        (x*scale_x, y*scale_y, x2*scale_x, y2*scale_y)
                    )
                boxes = scaled_boxes

        if self.target_transform:
            boxes = self.target_transform(boxes)

        return image, boxes

    def _set_images(self):
        for path, dirs, files in os.walk(self.labels_dir):
            for file in files:
                if 'ellipseList' in file:
                    with open(str(self.labels_dir.joinpath(file)), 'r') as f:
                        lines = filter(None, f.read().split('\n'))
                        counter = 0
                        boxes = []
                        for line in lines:
                            if counter == 0:
                                self.images.append(f'{line}.jpg')
                                boxes = []
                                counter = -1
                                continue
                            if counter == -1:
                                counter = int(line)
                                continue
                            if counter > 0:
                                major_axis_radius, minor_axis_radius, angle, center_x, center_y, prob = map(float, filter(None, line.split(' ')))
                                # boxes.append((major_axis_radius, minor_axis_radius, angle, center_x, center_y))
                                boxes.append((
                                    center_x - minor_axis_radius,
                                    center_y - major_axis_radius,
                                    center_x + minor_axis_radius,
                                    center_y + major_axis_radius
                                ))
                                counter -= 1
                                if counter == 0:
                                    self.bboxes.append(boxes)
            break

    def _read_image(self, img_path):
        img = Image.open(img_path)
        img = img.convert("RGB")
        return img

    @staticmethod
    def get_ellipse_bb(x, y, major, minor, angle, radians: bool = True):
        """
        Compute tight ellipse bounding box.

        see https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse#88020
        """
        if not radians:
            angle = np.radians(angle)
        t = np.arctan(-minor / 2 * np.tan(angle) / (major / 2))
        [min_x, max_x] = sorted([x + major / 2 * np.cos(t) * np.cos(angle) -
                                 minor / 2 * np.sin(t) * np.sin(angle) for t in (t + np.pi, t)])
        t = np.arctan(minor / 2 * 1. / np.tan(angle) / (major / 2))
        [min_y, max_y] = sorted([y + minor / 2 * np.sin(t) * np.cos(angle) +
                                 major / 2 * np.cos(t) * np.sin(angle) for t in (t + np.pi, t)])
        return min_x, min_y, max_x, max_y
