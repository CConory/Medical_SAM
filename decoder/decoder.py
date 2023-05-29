import torch.nn as nn
import torch.nn.functional as F
import torch


class DownAndUp(nn.Module):
    def __init__(self,in_channels, out_channels):
       super(DownAndUp, self).__init__()
       temp = out_channels
       self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, temp, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
        )
    def forward(self, x):
     
        return self.conv1(x)


class Up(nn.Module):
    """Upscaling"""

    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
    
        return x
    

# inherit nn.module
class Model(nn.Module):
    def __init__(self,img_channels=3, n_classes=1):
       super(Model, self).__init__()

       self.img_channels = img_channels
       self.n_classes = n_classes
       self.maxpool = nn.MaxPool2d(kernel_size=2)
       self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

       self.up = Up()

       self.up_conv1 = DownAndUp(256, 128)
       self.up_conv2 = DownAndUp(128, 64)



       
    def forward(self, x):
        
        x = self.up(x)

        x = self.up_conv1(x)
        
        x = self.up(x)
        x = self.up_conv2(x)

        x = self.out_conv(x)
        
        #x19 = torch.sigmoid(x18)
        return x
