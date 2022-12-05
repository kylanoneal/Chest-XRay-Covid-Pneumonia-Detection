import torch
import torch.nn as nn
import cv2
import numpy as np


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()

        self.features = torch.nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(8, stride=4),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(3136, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def get_prediction(img_filepath):
    img_size = (128, 128)
    channels = 1

    # Load model state on cpu
    loaded_dict = torch.load("custom_model_state_dict_94.pt", map_location=torch.device('cpu'))
    cust_model = TNet()
    cust_model.load_state_dict(loaded_dict)

    # Perform pre-processing steps
    img = cv2.imread(img_filepath, 0)
    img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
    img = img.astype(np.float) / 255 * 2 - 1
    img = np.reshape(img, (1, channels, img_size[0], img_size[1]))
    img -= np.mean(img)
    img = torch.FloatTensor(img)

    # Get model output
    cust_model(img)
    output = cust_model(img)
    _, pred = torch.max(output, 1)

    if pred == 0:
        return "Normal"
    if pred == 1:
        return "Non-COVID Pneumonia"
    if pred == 2:
        return "COVID-19 Pneumonia"