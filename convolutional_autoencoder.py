import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x


def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))


if __name__ == '__main__':
    EPOCH = 5
    BATCH_SIZE = 64
    LR = 1e-3

    train_data = datasets.CIFAR10(
        root='datasets/cifar10',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )

    test_data = datasets.CIFAR10(
        root='datasets/cifar10',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )

    model = ConvAutoencoder().cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_epochs = 100

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        # Training
        for data in train_loader:
            images, _ = data
            images = images.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # Sample outputs
    output = model(images)
    images = images.numpy()

    output = output.view(BATCH_SIZE, 3, 32, 32)
    output = output.detach().numpy()

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck']


    # Original Images
    print("Original Images")
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True,
                                 figsize=(12, 4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])
    plt.show()

    # Reconstructed Images
    print('Reconstructed Images')
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True,
                             figsize=(12, 4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
        imshow(output[idx])
        ax.set_title(classes[labels[idx]])
    plt.show()
