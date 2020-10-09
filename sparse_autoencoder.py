import time

import torch
from torch import nn
import torch.utils.data as torch_data
import torchvision
import matplotlib.pyplot as plt

from vanilla_autoencoder import AutoEncoder


def sparse_loss_l1(model, images):
    model_children = list(model.children())
    loss = 0
    values = images
    for i in range(len(model_children)):
        values = torch.relu((model_children[i](values)))
        loss += torch.mean(torch.abs(values))
    return loss


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
        (1 - rho) / (1 - rho_hat)))


def sparse_loss_kl(model, images):
    model_children = list(model.children())
    loss = 0
    values = images
    for i in range(len(model_children)):
        values = torch.relu((model_children[i](values)))
        loss += kl_divergence(0.05, values)
    return loss


def fit(model, dataloader, optimizer, lambd):
    model.train()
    running_loss = .0
    counter = 0
    for step, (x, b_label) in enumerate(dataloader):
        counter += 1
        x = x.to(device)
        b_x = x.view(-1, 28 * 28)
        b_y = x.view(-1, 28 * 28)
        optimizer.zero_grad()
        encoded, decoded = model(b_x)
        # loss = nn.MSELoss()(decoded, b_y) + lambd * sparse_loss_l1(model, b_x)
        loss = nn.MSELoss()(decoded, b_y) + lambd * sparse_loss_kl(model, b_x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / counter
    return epoch_loss


if __name__ == '__main__':
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

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

    train_loader = torch_data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = AutoEncoder()

    # train and validate the autoencoder neural network
    train_loss = []
    start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1} of {EPOCH}")
        train_epoch_loss = fit(model, train_loader, optimizer, 0.001)
        train_loss.append(train_epoch_loss)
    end = time.time()

    print(f"{(end - start) / 60:.3} minutes")
    # save the trained model
    torch.save(model.state_dict(),
               f"result/sparse_autoencoder/sparse_ae{EPOCH}.pth")

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('result/sparse_autoencoder/loss.png')
    plt.show()
