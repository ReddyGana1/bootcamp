import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


                    

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))

])

train_data=datasets.MNIST(root='./dir',download=True,transform=transform)
test_data=datasets.MNIST(root='./dir',download=False,transform=transform)

train_loader=DataLoader(train_data,batch_size=128,shuffle=True)
test_loader=DataLoader(test_data,batch_size=128,shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        #encoder
        self.encoder=nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1),
            torch.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),
            torch.ReLU()
        )

        #decoder
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2,output_padding=1),
            torch.ReLU(),
            nn.ConvTranspose2d(16,1,3,stride=2,output_padding=1),
            torch.tanh()
        )

        def forward(self,x):
            x=self.encoder(x)
            x=self.decoder(x)
            return x
        


model=Autoencoder()
criterian=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

def add_noise(img):
    noisy=torch.randn_like(img)*0.5
    noisy=image+noisy
    noise=torch.clamp(noisy,-1.,1.)

for epoch in range(20):
    for image,label in train_loader:
        noisy_image=add_noise(image)
        output=model(noisy_image)
        loss=criterian(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
correct=0.0
total=0.0
model.eval()
with torch.no_grad():
    for image,label in test_loader:
        output=model(image)
        max,predicted=torch.max(output,1)
        correct+= (predicted==label).sum().item()
        total+=label.size(0)
    print(f"test accuracy:{(correct/total)*100}")