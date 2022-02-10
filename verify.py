# -*- coding: UTF-8 -*-
import os
import pickle
import time
from typing import Tuple

import matplotlib.pyplot as plt
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

    def get_bounds(self):
        return self.based_polygon.bounds

    def get_centroid(self):
        return self.based_polygon.centroid

    def intersect(self, point: Tuple[float, float]):
        return Point(point).within(self.based_polygon)



class ValidationSet(Dataset):

    def __init__(self):
        return

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        path = r"/home/lennit/Desktop/Kurs/Output/"  # I know thats bad
        image = Image.open(os.path.join(path, f'captcha_{index}_val.png'))

        image = BasicDataset.preprocess(image, scale=0.5, is_mask=False)

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),  # passing back the image
            'polygon': os.path.join(path, f'captcha_{index}_val.polygon')  # passing back the path to poly file as label
        }


class Counter:

    def __init__(self):
        self.counter = 0

    def increment(self):
        self.counter += 1

    def print(self):
        print(f'Function triggered {self.counter} times')


def get_centroid(mask):
    #counter.increment()
    highest = 0
    lowest = 0
    width = 0

    height = 0
    for row in mask:
        if row.any() == 1:
            for i in range(len(row)):
                if row[i] == 1:
                    highest = height
                    while not mask[height, i] == 0:
                        height += 1

                    lowest = height
                    width = i
                    break
            break
        height += 1

    centroid = highest + (lowest - highest) / 2
    return width * 2, centroid * 2


def get_centroid_center_of_gravity(mask):
    n_one = 0
    row_sum = 0
    column_sum = 0

    h = 0
    for row in mask:
        h += 1
        if row.any() == 1:
            for column in range(len(row)):
                if row[column] == 1:
                    n_one += 1
                    row_sum += h
                    column_sum += column

    centroid = (column_sum / n_one) * 2, (row_sum / n_one) * 2
    if not mask[int(centroid[1] / 2)][int(centroid[0] / 2)] == 1:
        centroid = get_centroid(mask)

    return centroid


if __name__ == "__main__":
    start = time.time()
    batch_size = 1
    n_correct = 0
    n_samples = 0

    device = torch.device('cuda')
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net.load_state_dict(torch.load('/home/lennit/Desktop/Kurs/Pytorch-UNet/checkpoint_epoch17.pth',
                                   map_location=device))
    net.to(device)
    net.train()
    # net.eval()

    dataset = ValidationSet()
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    counter = Counter()

    step = 0
    for batch in data_loader:
        step += 1
        images = batch['image']
        polygons = batch['polygon']

        images = images.to(device=device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=True):
            pred = net(images)

            for i in range(batch_size):
                output = pred[i]

                mask = torch.softmax(pred, dim=1).argmax(dim=1).float().cpu()[0]
                mask = mask.cpu().numpy()
                centroid = get_centroid_center_of_gravity(mask)

                verify_instance = Verify(polygons[i])
                if not verify_instance.intersect(centroid):
                    print(f'{polygons} failed.')
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(images[i].cpu().permute(1, 2, 0).numpy().astype('uint8'))
                    ax[1].imshow(mask)
                    plt.savefig(os.path.join("fails3", f"captcha_{step - 1}_val.png"))

                else:
                    n_correct += 1
                n_samples += 1

        if step % 100 == 0:
            print(f'{step}/10000: current accuracy {n_correct/n_samples * 100}%')
    end = time.time()
    print('Testing finished.')
    print(f'Process took {(end - start) / 60} min, that\'s an average of {(end - start) /10000} seconds per image!')
    print(f'Result: {n_correct} of {n_samples} correct -> {n_correct/n_samples * 100}%')
    counter.print()
