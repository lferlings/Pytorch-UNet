# -*- coding: UTF-8 -*-
import pickle
from typing import Tuple

import numpy
from PIL import Image

from shapely.geometry import Polygon, Point

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn.functional as F

import predict
from unet import UNet
from utils.data_loading import BasicDataset


class Verify:
    based_polygon: Polygon

    def __init__(self, polygon_path: str):
        with open(polygon_path, 'rb') as file:
            points = pickle.load(file)
            self.based_polygon = Polygon(points)

    def __eq__(self, other):
        if isinstance(other, Verify):
            if other.center_xy[0] == self.center_xy[0] and other.center_xy[1] == self.center_xy[1] and other.radius == self.radius:
                return True
            else:
                return False

    def intersect(self, point: Tuple[float, float]):
        return Point(point).within(self.based_polygon)


def get_centroid(mask):
    highest = ()

    height = 0
    for row in mask[1]:
        if row.any() == 0:
            for i in range(len(row)):
                width = 0
                if i == 1:
                    highest = (height, width)

                    while not mask[height, width] == 0:
                        height += 1

                    lowest = (height, width)

                width += 1
        height += 1



def get_images(batch, batch_size):
    images = []
    path = "/home/lennit/Desktop/Kurs/Validation/captcha_"
    for i in range(batch_size):
        img = Image.open(path + str(batch * 10 + i) + "_val.png")
        images.append(img)
    return images

def get_polygons(batch, batch_size):
    polygons = []
    path = "/home/lennit/Desktop/Kurs/Validation/captcha_"
    for i in range(batch_size):
        poly = Image.open(path + str(batch * 10 + i) + "_val.polygon")
        polygons.append(poly)
    return polygons


def verify(net, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()



    with torch.no_grad():
        n_samples = 0
        n_correct = 0
        batches = 10000 / batch_size

        for batch in range(batches):
            images = get_images(batch, batch_size)
            pred = net(images)

            for i in range(batch_size):
                centroid = get_centroid(pred[i])
                v = Verify(f'/home/lennit/Desktop/Kurs/Validation/captcha_{batch * 10 + i}_val.polygon')

                if not v.intersect(centroid):
                    print(f'captcha_{batch * 10 + i} wurde nicht erkannt.')
                else:
                    n_correct += 1
                n_samples += 1

if __name__ == "__main__":
    device = torch.device('cuda')
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net.load_state_dict(torch.load(r'C:\Users\lenna\sciebo\Semester 5\Captcha\Workspace\Pytorch-UNet\checkpoints\checkpoint_epoch7.pth',
                                   map_location=device))
    net.to(device)
    image = Image.open(r'C:\Users\lenna\sciebo\Semester 5\Captcha\CaptchaVAL\Output\captcha_0_val.png')

    mask = predict.predict_img(net, image, device)
    print(get_centroid(mask))

        # if net.n_classes > 1:
        #     probs = F.softmax(output, dim=1)[0]
        # else:
        #     probs = torch.sigmoid(output)[0]
        #
        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((image.size[1], image.size[0])),
        #     transforms.ToTensor()
        # ])
        #
        # full_mask = tf(probs.cpu()).squeeze()
        # print(full_mask)
        # mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
        # zero = mask.any() == 0
