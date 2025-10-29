import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#transforms
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])  

#load the data
train_data= datasets.CIFAR100(root='./dir', train=True, download=True, transform=transform)
test_data= datasets.CIFAR100(root='./dir', train=False, download=True, transform=transform)
train_loader= DataLoader(train_data, batch_size=64, shuffle=True)
test_loader= DataLoader(test_data, batch_size=64, shuffle=False)

#architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 100)


    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return x
    
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#train the model
epoch_loss = 0.00
for epoch in range(3): 
    for images, labels in train_loader:
        op=model(images)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()    
        epoch_loss += loss.item()
    print(f"epoch loss:{epoch_loss}")

#test the model
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()