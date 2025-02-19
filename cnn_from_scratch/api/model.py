import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

class CustomNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 12, 5) #input channel, output channel, size of kernel, 28-5(kernel) / stride (1) + 1 or (12, 24, 24)
        self.pool = nn.MaxPool2d(2, 2) #2*2 pixels and creates 1 pixel out of it (12, 12, 12)
        self.conv2 = nn.Conv2d(12, 24, 5) # 12-5 = 7 + 1 = 8 (24, 8,  8) -> (24, 4, 4) defined in fw logic -> flatten 24 * 4 * 4
        self.fc1 = nn.Linear(24 * 4 * 4, 120) #fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #the classes, 10 in this case
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #relu activation function, breaks linearity
        x = self.pool(F.relu(self.conv2(x))) 
        x = torch.flatten(x, 1) #flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = CustomNet()
loss_function = nn.CrossEntropyLoss() #categorical
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

torch.save(net.state_dict(), 'trained_net.pth')

correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in testloader:
        
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#87% on two epochs
        
