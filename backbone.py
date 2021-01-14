import torch
from torch import nn
import torch.nn.functional as F

class simple_backbone(nn.Module):
    def __init__(self, in_planes, out_planes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=2**3, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(in_channels=2**3, out_channels=2**4, kernel_size=3, bias=True)
        self.conv3 = nn.Conv2d(in_channels=2**4, out_channels=2**5, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(in_channels=2**5, out_channels=2**6, kernel_size=3, bias=True)
        self.conv5 = nn.Conv2d(in_channels=2**6, out_channels=out_planes, kernel_size=3, bias=True)

        self.activation = nn.functional.relu

        self.norm1 = nn.BatchNorm2d(2**3, 1e-3, 1e-3)
        self.norm2 = nn.BatchNorm2d(2**4, 1e-3, 1e-3)
        self.norm3 = nn.BatchNorm2d(2**5, 1e-3, 1e-3)
        self.norm4 = nn.BatchNorm2d(2**6, 1e-3, 1e-3)

    def forward(self, x, mask) :
        x = self.conv1(x)
        x = self.activation(x)
        x = self.norm1(x)
        m = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

        x = self.conv2(x)
        x = self.activation(x)
        x = self.norm2(x)
        m = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

        
        x = self.conv3(x)
        x = self.activation(x)
        x = self.norm3(x)
        m = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]


        x = self.conv4(x)
        x = self.activation(x)
        x = self.norm4(x)
        m = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]


        x = self.conv5(x)
        output = self.activation(x)
        m = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]


        return output, m