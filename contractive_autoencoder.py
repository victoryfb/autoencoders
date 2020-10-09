import os

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        self.fc1 = nn.Linear(784, 400, bias=False)  # Encoder
        self.fc2 = nn.Linear(400, 784, bias=False)  # Decoder

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 784)))
        return h1

    def decoder(self, z):
        h2 = self.sigmoid(self.fc2(z))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2

    # Writing data in a grid to check the quality and progress
    def samples_write(self, x, epoch):
        _, samples = self.forward(x)
        samples = samples.data.cpu().numpy()[:16]
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        if not os.path.exists('result/contractive_autoencoder'):
            os.makedirs('result/contractive_autoencoder')
        plt.savefig(
            f'result/contractive_autoencoder/{str(epoch).zfill(3)}.png',
            bbox_inches='tight')
        plt.close(fig)


def loss_function(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss
    Evaluate the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.

    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regularization term
    Returns:
        Variable: the (scalar) CAE loss
    """
    mse = mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h)  # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(W ** 2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


def train(epoch, lambd):
    model.train()
    train_loss = 0

    for idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        hidden_representation, recons_x = model(data)

        # Get the weights
        W = model.state_dict()['fc1.weight']
        loss = loss_function(W, data.view(-1, 784), recons_x,
                             hidden_representation, lambd)

        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if idx % 100 == 0:
            print('Train epoch: {} [{}/{}({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                       100 * idx / len(train_loader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    model.samples_write(data, epoch)


if __name__ == '__main__':
    EPOCH = 5
    BATCH_SIZE = 64
    LR = 1e-3

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    train_data = datasets.MNIST(
        root='datasets/mnist',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = torch_data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )

    test_data = datasets.MNIST(
        root='datasets/mnist',
        train=False,
        transform=transforms.ToTensor()
    )

    test_loader = torch_data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )

    mse_loss = nn.BCELoss(reduction='sum')
    model = CAE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(EPOCH):
        train(epoch, 1e-4)
