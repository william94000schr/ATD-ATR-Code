# @author : Grégoire Mallet
# This file implements a GAN method, used here to increase the amount of data of a dataset (specifically, the MSTAR dataset ;)
# Imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image

#First, loading the dataset : we need the data with the following structure (you can adapt),
# where each class is in a specific folder : 
# data_MSTAR
#  -> 2S1 (it's a class)
#       -> img_1.JPG
#       -> img_2.JPG
#       -> ... (all the images of this class)
#  -> all the classes...

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(128), #Not necessarly a good idea to use it
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

dataset = datasets.ImageFolder(root="../data/images/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#Then, we define our GAN, which is composed of a generator and a discriminator.
#Beginning with the generator : 
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,512,4,1,0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4,2,1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128,4,2,1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


#And then the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64,4, 2, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64, 128,4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,256, 4, 2, 1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256,1, 4,1 , 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)  # sortie de taille [batch, 1, 13, 13]


#Now, the training : it can be quite long, better use a gpu (I used personnaly a CPU).
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim =100
generator =Generator(latent_dim).to(device)
discriminator= Discriminator().to(device)

criterion=nn.BCELoss()
optimizer_g=optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) #Values suggered by Claude, I didn't try to optimize them :)
optimizer_d=optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs =3 #More with a GPU

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        print(i)
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        #We train the discriminator
        optimizer_d.zero_grad()
        
        #print(real_imgs)
        #Real or false predictions
        real_pred = discriminator(real_imgs)
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_imgs = generator(z)
        fake_pred = discriminator(fake_imgs.detach())

        valid = torch.ones_like(real_pred,device=device)
        fake = torch.zeros_like(fake_pred,device=device)

        # Loss calculus
        real_loss=criterion(real_pred,valid)
        fake_loss=criterion(fake_pred,fake)
        d_loss=(real_loss+fake_loss)/2
        d_loss.backward()
        optimizer_d.step()

        #Generator training
        optimizer_g.zero_grad()
        gen_pred = discriminator(fake_imgs)
        valid= torch.ones_like(gen_pred, device=device)
        g_loss = criterion(gen_pred,valid)
        g_loss.backward()
        optimizer_g.step()
        
        
    print(f"[{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")



#Here, I visualize some generated images. You can try to save them or other...
z = torch.randn(16, latent_dim, 1, 1, device=device)
fake_imgs = generator(z).detach().cpu()
grid = utils.make_grid(fake_imgs, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()

#Saving the image
for i, img in enumerate(fake_imgs):
    save_image(img, f"../experiments/outputs/generated_{i+1:03d}.png", normalize=True)