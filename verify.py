# -*- coding: UTF-8 -*-
import os
import pickle
from typing import Tuple

import numpy
from PIL import Image

from shapely.geometry import Polygon, Point

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
from tqdm import tqdm

import predict
from utils import data_loading
from unet import UNet
from utils.data_loading import BasicDataset
from utils import utils


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


class ValidationSet(Dataset):

    def __init__(self):
        return

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        path = r"/home/lennit/Desktop/Kurs/CaptchaVAL/Output/"  # I know thats bad
        image = Image.open(os.path.join(path, f'captcha_{index}_val.png'))

        image = BasicDataset.preprocess(image, scale=0.5, is_mask=False)

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),  # passing back the image
            'polygon': os.path.join(path, f'captcha_{index}_val.polygon')  # passing back the path to poly file as label
        }



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




if __name__ == "__main__":
    batch_size = 2
    device = torch.device('cuda')
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net.load_state_dict(torch.load('/home/lennit/Desktop/Kurs/Pytorch-UNet/checkpoint_epoch8.pth',
                                   map_location=device))
    net.to(device)

    dataset = ValidationSet()
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    for batch in data_loader:
        images = batch['image']
        polygons = batch['polygon']

        images = images.to(device=device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=True):
            pred = net(images)

            for i in range(batch_size):
                output = pred[i]

                if net.n_classes > 1:
                    probs = F.softmax(output, dim=1)[0]
                else:
                    probs = torch.sigmoid(output)[0]

                tf = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((1000, 1000)),
                    transforms.ToTensor()
                ])

                full_mask = tf(probs.cpu()).squeeze()
                image = tf(images[i].cpu().cpu()).squeeze()
                mask = (full_mask.argmax(dim=0)).numpy()
                # if net.n_classes == 1:
                #     mask = (full_mask > 0.5).numpy()
                # else:
                #     mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes).numpy()

                utils.plot_img_and_mask(image, mask)



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
