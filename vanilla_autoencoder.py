import torch
from torch import nn
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)  # compress to 3 features to visualize
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # compress to a range (0, 1)
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

    def optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


if __name__ == '__main__':
    EPOCH = 5
    BATCH_SIZE = 64
    LR = 0.005
    N_TEST_IMG = 5

    train_data = torchvision.datasets.MNIST(
        root='datasets/mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_data = torchvision.datasets.MNIST(
        root='datasets/mnist',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_loader = torch_data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = torch_data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    autoencoder = AutoEncoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))

    view_data = train_data.data[:N_TEST_IMG].view(-1, 28 * 28).type(
        torch.FloatTensor) / 255

    for epoch in range(EPOCH):
        for step, (x, b_label) in enumerate(train_loader):
            b_x = x.view(-1, 28 * 28)
            b_y = x.view(-1, 28 * 28)

            encoded, decoded = autoencoder(b_x)

            loss = loss_func(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch: ', epoch,
                      '| train loss: %.4f' % loss.data.numpy())
                _, decoded_data = autoencoder(view_data)
                f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
                for i in range(N_TEST_IMG):
                    a[0][i].imshow(
                        np.reshape(view_data.data.numpy()[i], (28, 28)),
                        cmap='gray')
                    a[0][i].set_xticks(())
                    a[0][i].set_yticks(())
                    a[1][i].imshow(
                        np.reshape(decoded_data.data.numpy()[i], (28, 28)),
                        cmap='gray')
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())
                plt.savefig(f'result/vanilla_autoencoder/{epoch}_{step}.png')
                plt.close(f)

    view_data = train_data.data[:200].view(-1, 28 * 28).type(
        torch.FloatTensor) / 255
    encoded_data, _ = autoencoder(view_data)
    fig = plt.figure(2)
    ax = Axes3D(fig)
    X = encoded_data.data[:, 0].detach().numpy()
    Y = encoded_data[:, 1].detach().numpy()
    Z = encoded_data[:, 2].detach().numpy()
    values = train_data.targets[:200].numpy()
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.savefig('result/vanilla_autoencoder/res_autoencoder.png')
