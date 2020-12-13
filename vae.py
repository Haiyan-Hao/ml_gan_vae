import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class vae_linear(nn.Module):
    def __init__(self):
        super(vae_linear, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, 20)
        self.fc32 = nn.Linear(256, 20)
        self.fc4 = nn.Linear(20, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='mean')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for idx, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_x, mu, log_var = model(x)
        loss = loss_function(recon_x, x, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Train loss: %.4f' % (train_loss/ len(train_loader)))
    return train_loss/len(train_loader)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for  _, (x, _) in enumerate(test_loader):
            recon_x, mu, log_var = model(x)
            test_loss += loss_function(recon_x, x, mu, log_var).item()
    print('Test set loss: %.4f' % (test_loss/len(test_loader)))
    return test_loss/len(test_loader)

def generate_random_imgs(model):
    rand_var = torch.randn(32, 20)
    fake_imgs = model.decoder(rand_var).reshape(-1, 28, 28).unsqueeze(1).detach()
    grid_img=torchvision.utils.make_grid(fake_imgs, nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0))

if __name__ == "__main__":
    batchsz = 32
    train_data = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=False)
    test_data = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchsz)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batchsz)

    # show the original data:
    _, (examples, _) = next(iter(enumerate(test_loader)))
    grid_img=torchvision.utils.make_grid(examples, nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0))

    # create models and train
    model = vae_linear()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    out = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss'])
    for epoch in range(100):
        train_loss = train(model, train_loader, optimizer)
        test_loss = test(model, test_loader)
        out = out.append({'epoch':epoch, 'train_loss': train_loss, 'test_loss':test_loss}, ignore_index=True)

    pickle.dump(model, open('./vae_model.p', 'wb'))
    out.to_csv('./performance_vae.csv', index=False)

    # create fake images
    generate_random_imgs(model)

