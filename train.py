import numpy as np
import torch
from torch import dtype, nn 
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
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model = simple_transNet(10,2048,4,6,6,"cuda:0")
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 30
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output_dir = '/workspace/Simple-Transformer/history_output_model'
    for epoch in tqdm(range(epochs)):
        with tqdm(total=len(loader)) as bar:
            for i,data in enumerate(loader):
                imgs, targets = data[0],data[1]
                imgs = imgs.cuda().to(dtype=torch.float32)
                targets = targets.cuda()
                optimizer.zero_grad()
                output = model(imgs)
                res = output[1]
                loss = loss_fn(res, targets)
                loss.backward()
                optimizer.step()
                bar.set_description('In {} epoch progress(loss:{}):'.format(epoch, loss))
                bar.update(1)
            if epoch % 1 == 0:
                checkpoint_path = output_dir+f'/checkpoint{epoch:04}.pth'
                state = {'net':model.state_dict, 'opt':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state, checkpoint_path)
        

