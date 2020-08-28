# %% [IMPORT MODULE]
from os import listdir
import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# %% [Some Constants]
# LOC = "../data/test/"
DIM = 256

NAME_TO_INT = {'omphalotus_olearius': 0, 'amanita_caesarea': 1,
               'cantharellus_cibarius': 2,  'volvariella_volvacea': 3,  'amanita_phalloides': 4}
INT_TO_NAME = {0: 'omphalotus_olearius', 1: 'amanita_caesarea',
               2: 'cantharellus_cibarius', 3: 'volvariella_volvacea', 4: 'amanita_phalloides'}

MEAN_LOC = os.path.join(os.path.abspath(
    os.path.dirname('__file__')), "app/models/rgb_mean.txt")
STD_LOC = os.path.join(os.path.abspath(
    os.path.dirname('__file__')), "app/models/rgb_std.txt")
MODEL_LOC = os.path.join(os.path.abspath(
    os.path.dirname('__file__')), "app/models/test_net.pt")


# %% [AlexNet]


class AlexNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=5):
        super(AlexNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=96,
                      kernel_size=(11, 11), stride=4, padding=0),  # 0
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=(5, 5), stride=1, padding=2),  # 4
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),  # 8
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),  # 10
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),  # 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.AvgPool2d(kernel_size=(6, 6))
        )
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=256, out_features=4096),  # 0
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),  # 2
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=out_chans)  # 4
        )

    def forward(self, img):
        img = self.conv_block(img)
        img = img.view(img.shape[0], -1)
        img = self.fc_block(img)
        return img

    def length_(self, length, p, f, s):
        return math.floor(((length+2*p-f)/s)+1)

    def width_(self, width, p, f, s):
        return math.floor(((width+2*p-f)/s)+1)

    def padding_same(self, x, s, f):
        return ((s*x-x)/2)+math.floor(f/2)

    def init_weight(self):
        conv_idx = [0, 4, 8, 10, 12]
        lin_idx = [0, 2, 4]
        for idx in conv_idx:
            torch.nn.init.kaiming_normal_(self.conv_block[idx].weight)
        for idx in lin_idx:
            torch.nn.init.kaiming_normal_(self.fc_block[idx].weight)


model = AlexNet()
model.load_state_dict(torch.load(MODEL_LOC, map_location='cpu'))

# %% [LOAD MEAN & STD]
with open(MEAN_LOC, 'r') as f:
    mean = list(map(lambda x: float(x), f.readline().split(" ")))
with open(STD_LOC, 'r') as f:
    std = list(map(lambda x: float(x), f.readline().split(" ")))
