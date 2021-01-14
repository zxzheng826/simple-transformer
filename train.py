import numpy as np
import torch
from torch import nn 
from torch import utils
from torch import optim
from torch.utils.data import DataLoader
import torch.functional as F 
import torchvision
import cv2
from tqdm import tqdm

from simple_transNet import simple_transNet, post_process

def PIL2Numpy(PIL_image):
    return np.array(PIL_image)


if __name__ == '__main__':
    train_dataset = torchvision.datasets.MNIST("./datasets",train=True,transform=PIL2Numpy, target_transform=PIL2Numpy, download=True)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = simple_transNet(10,2048,4,6,6,"cuda:0")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 30
    for epochs in tqdm(range(epochs)):
        for i,data in enumerate(loader):
            imgs, targets = data[0],data[1]
            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

