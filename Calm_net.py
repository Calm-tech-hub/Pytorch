import torch.nn as nn
import torch


class Calm(nn.Module):
    def __init__(self):
        super(Calm,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,padding = 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding = 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding = 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16,64),
            nn.Linear(64,10)
        )
    def forward(self,input):
        output = self.model(input)
        return output

calm = Calm()
input = torch.randn((64,3,32,32))
output = calm(input)
print(output.shape) 
