from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision.ops.boxes import batched_nms, box_iou

from .models import PNet, RNet
from .mtcnn_utils import *


class LiteMTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pnet = PNet()
        self.rnet = RNet()
        self.minsize = 20
        self.factor = 0.709
        self.iou_threshold_0 = 0.5
        self.iou_threshold_1 = 0.5
        self.iou_threshold_2 = 0.1
        self.threshold_pnet = 0.97
        self.threshold_rnet = 0.97
        self.image_size = (100, 100)
        self.boxes = None

    def detect(self, image: Image):
        imgs = np.stack([np.uint8(image)])
        imgs = torch.as_tensor(imgs.copy())

        model_dtype = next(self.pnet.parameters()).dtype

        # changing batch size to [batch, channels, height, width)
        imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

        batch_size = len(imgs)
        h, w = imgs.shape[2:4]
        m = 12.0 / self.minsize
        minl = min(h, w)
        minl = minl * m

        # Create scale pyramid
        scale_i = m
        scales = []
        while minl >= 12:
            scales.append(scale_i)
            scale_i = scale_i * self.factor
            minl = minl * self.factor

        # First stage
        boxes = []
        image_inds = []

        scale_picks = []

        offset = 0
        for scale in scales:
            im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
            im_data = (im_data - 127.5) * 0.0078125
            reg, probs = self.pnet(im_data)

            boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, self.threshold_pnet)
            boxes.append(boxes_scale)
            image_inds.append(image_inds_scale)

            pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, self.iou_threshold_0)
            scale_picks.append(pick + offset)
            offset += boxes_scale.shape[0]

        boxes = torch.cat(boxes, dim=0)
        image_inds = torch.cat(image_inds, dim=0)

        scale_picks = torch.cat(scale_picks, dim=0)

        # NMS within each scale + image
        boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, self.iou_threshold_1)
        boxes, image_inds = boxes[pick], image_inds[pick]

        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        qq1 = boxes[:, 0] + boxes[:, 5] * regw
        qq2 = boxes[:, 1] + boxes[:, 6] * regh
        qq3 = boxes[:, 2] + boxes[:, 7] * regw
        qq4 = boxes[:, 3] + boxes[:, 8] * regh
        boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
        boxes = rerec(boxes)
        y, ey, x, ex = pad(boxes, w, h)

        # Second stage
        if len(boxes) > 0:
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (24, 24)))
            im_data = torch.cat(im_data, dim=0)

            im_data = (im_data - 127.5) * 0.0078125

            out = self.rnet(im_data)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            score = out1[1, :]
            ipass = score > self.threshold_rnet
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            # NMS within each image
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, self.iou_threshold_2)
            boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
            boxes = bbreg(boxes, mv)
            boxes = rerec(boxes)
            return boxes

    def forward(self, image: Image):
        self.boxes = self.detect(image)
        if self.boxes is None:
            return []
        images = []
        indexes = []
        i = 0
        for x, y, x2, y2, prob in self.boxes:
            box = (x.item(), y.item(), x2.item(), y2.item())
            indexes.append((i, box[0]))
            images.append(image.crop(box).copy().resize((self.image_size[0], self.image_size[1]), Image.BILINEAR))
            i += 1
        if i > 1:
            indexes.sort(key=lambda arg: arg[1])
            images = [images[idx] for idx, _ in indexes]
        return images
