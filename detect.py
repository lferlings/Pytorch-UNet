import verify
from unet import UNet
from utils.data_loading import BasicDataset

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import os
import sys

class DetectSet(Dataset):

    def __init__(self):
        super(DetectSet, self).__init__()
        try:
            self.item = sys.argv[1]
        except IndexError:
            print("Usage: python detect.py <index>")
            exit()


    def __len__(self):
        return 1

    def __getitem__(self, index):
        path = r"/home/lennit/Desktop/Kurs/Output/"  # Path to captcha folder
        image = Image.open(os.path.join(path, f'captcha_{self.item}_val.png'))

        image = BasicDataset.preprocess(image, scale=0.5, is_mask=False)

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),  # passing back the image
            'polygon': os.path.join(path, f'captcha_{self.item}_val.polygon')  # passing back the path to poly file as label
        }

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net.load_state_dict(torch.load('/home/lennit/Desktop/Kurs/Pytorch-UNet/checkpoint_epoch17.pth',
                                   map_location=device))
    net.to(device)
    net.train()

    dataset = DetectSet()
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    if not os.path.isdir("customplots"):
        os.mkdir("customplots")

    for batch in data_loader:
        image_as_list = batch['image']
        image_as_list = image_as_list.to(device=device, dtype=torch.float32)
        pred = net(image_as_list)
        output = pred[0]

        mask = torch.softmax(pred, dim=1).argmax(dim=1).float().cpu()[0]
        mask = mask.cpu().numpy()
        centroid = verify.get_centroid_center_of_gravity(mask)

        verify = verify.Verify(batch['polygon'][0])
        print(f"Check result: {verify.intersect(centroid)}")
        print(f"Predicted point: {centroid}")
        print(f"Real centroid: {verify.get_centroid()}")
        print(f"Bounds: {verify.get_bounds()}")
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_as_list[0].cpu().permute(1, 2, 0).numpy().astype('uint8'))
        ax[1].imshow(mask)
        plt.savefig(os.path.join("customplots", f"captcha_{sys.argv[1]}_val.png"), dpi=300, transparent=True)
        plt.show()
