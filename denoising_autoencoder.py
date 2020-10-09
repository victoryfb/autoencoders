import time

import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.utils.data as torch_data
import matplotlib.pyplot as plt

from vanilla_autoencoder import AutoEncoder


def add_noise(inputs):
    return inputs + torch.randn(inputs.size()) * 0.2


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor: min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor: tensor_round(tensor))
])


def fit(model, dataloader, optimizer):
    model.train()
    running_loss = .0
    counter = 0
    for step, (x, b_label) in enumerate(dataloader):
        counter += 1
        b_x = x.view(-1, 28 * 28)
        b_y = x.view(-1, 28 * 28)
        noisy_b_x = add_noise(b_x)

        optimizer.zero_grad()
        encoded, decoded = model(noisy_b_x)
        loss = nn.BCELoss()(decoded, b_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / counter
    return epoch_loss


if __name__ == '__main__':
    EPOCH = 5
    BATCH_SIZE = 64
    LR = 1e-3

    train_data = torchvision.datasets.MNIST(
        root='datasets/mnist',
        train=True,
        transform=img_transform,
        download=True
    )

    train_loader = torch_data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = AutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # train and validate the autoencoder neural network
    train_loss = []
    start = time.time()
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1} of {EPOCH}")
        train_epoch_loss = fit(model, train_loader, optimizer)
        train_loss.append(train_epoch_loss)
    end = time.time()

    print(f"{(end - start) / 60:.3} minutes")
    # save the trained model
    torch.save(model.state_dict(),
               f"result/denoising_autoencoder/sparse_ae{EPOCH}.pth")

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('result/denoising_autoencoder/loss.png')
    plt.show()