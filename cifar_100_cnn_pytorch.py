#import torch
#import torch.nn as nn
#import torch.optim
#from torchvision.datasets import datasets, transforms
#from torch.util.data import DataLoader

#transform = transforms.Compose([
#    transforms.ToTensor(),
    # FashionMNIST is single-channel (1). Normalize must match the number of channels.
#    transforms.Normalize((0.5,), (0.5,))
#])

#train_data = datasets.CIFAR100(root='./dir', train=True, download=True, transforms=transforms)
#test_data = datasets.CIFAR100(root='./dir', train=False, download=False, transforms=transforms)

#train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
#test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

#class cnn_cifar100(nn.Module):
#    def _init_(self):
#        super(cnn_cifar100, self)._init_()
#        self.conv1 = nn.Conv2d(3, 32, 5) #Input channels=3, output channels=32, kernel_size=3, padding=1; output channels is based on the number of filters used, one each per filter
#        # Asyou move further in the convolution, the numebr of layers must increase but the size of the filter mist reduce
#        # 3 conclusions --- (1) no. of channels = no. of filters; (2) as the depth increases, the number of filters increases; (3) as the depth increases, the size of filters decreases
#        # kernel_size decreases as depth increases because as the depth increases, the image size decreases
#        self.pool1 = nn.MaxPool2d(2, 2) #Usually take 2x2 filter size, no filter values

#        self.conv2 = nn.Conv2d(32, 64, 3) #maxpool reduces image, channels remain same
#        self.pool2 = nn.MaxPool2d(2, 2)

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

train_data = datasets.FashionMNIST(root='./dir', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./dir', train=False, download=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
class cnn_Fashionmnist(nn.Module):
    def _init_(self):
        super(cnn_Fashionmnist, self)._init_()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1) # input channels = 1 for FashionMNIST
        # As you move further in the convolution, the number of layers must increase but the size of the filter must reduce
        # 3 conclusions --- (1) no. of channels = no. of filters; (2) as the depth increases, the number of filters increases; (3) as the depth increases, the size of filters decreases
        # kernel_size decreases as depth increases because as the depth increases, the image size decreases
        self.pool1 = nn.MaxPool2d(2, 2) #Usually take 2x2 filter size

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # maxpool reduces image, channels increase as set by conv
        self.pool2 = nn.MaxPool2d(2, 2)
        # If we use the pooling layer (2, 2) the size is halved, the number of channels does not change
        # 28x28x1 -> 14x14x32
        # 14x14x64 -> 7x7x64; 7x7 image size, 64 channels

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*64, 256) #Number of inputs here is the number of pixels after conv/pool
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.flatten(1) 
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.relu(x)

        x = self.fc4(x)
        x = torch.relu(x)

        x = self.fc5(x)
        return x
    
model = cnn_Fashionmnist()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(20):
    for image, label in train_loader:
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Test  
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total}") 
        