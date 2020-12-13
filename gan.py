import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 784)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    # forward method
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


def train_G():
    G.zero_grad()
    G_output = G(torch.randn(batchsz, 64))
    D_output = D(G_output)
    G_loss = criterion(D_output, torch.ones(batchsz, 1))
    G_loss.backward()
    G_optimizer.step()
    return G_loss.item()

def train_D(x):
    D.zero_grad()
    x_real, y_real = x.view(-1, 784), torch.ones(batchsz, 1)
    x_fake, y_fake = G(torch.randn(batchsz, 64)), torch.zeros(batchsz, 1)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    return D_loss.item()

def generate_random_imgs(Generator):
    rand_var = torch.randn(32, 64)
    fake_imgs = Generator(rand_var).reshape(-1, 28, 28).unsqueeze(1).detach()
    grid_img=torchvision.utils.make_grid(fake_imgs, nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0))

if __name__ == "__main__":
    batchsz = 32
    train_data = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=False)
    test_data = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchsz)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batchsz)

    # create models and train
    G = Generator()
    D = Discriminator()
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=0.001)
    D_optimizer = optim.Adam(D.parameters(), lr=0.001)

    out = pd.DataFrame(columns=['epoch', 'D_loss', 'G_loss'])
    for epoch in range(50):
        D_loss = 0
        G_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            D_loss += train_D(x)
            G_loss += train_G()
        out = out.append({'epoch':epoch, 'D_loss':D_loss/len(train_loader), 'G_loss':G_loss/len(train_loader)}, ignore_index=True)
        print('%d: loss_d: %.3f, loss_g: %.3f' % ((epoch), D_loss/len(train_loader), G_loss/len(train_loader)))

    pickle.dump(D, open('./gan_D.p', 'wb'))
    pickle.dump(G, open('./gan_G.p', 'wb'))
    out.to_csv('./performance_gan.csv', index=False)

    # create fake images
    generate_random_imgs(G)

