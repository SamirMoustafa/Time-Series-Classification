import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fastprogress.fastprogress import master_bar, progress_bar

from src.utils import TimeSeriesDataset


class Encoder(nn.Module):
    def __init__(self, capacity, latent_dims, scalable_dim):
        super(Encoder, self).__init__()
        c = capacity
        self.scalable_dim = scalable_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(in_features=c * self.scalable_dim, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c * self.scalable_dim, out_features=latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)

        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, capacity, latent_dims, scalable_dim):
        super(Decoder, self).__init__()
        self.c = capacity
        self.scalable_dim = scalable_dim - 3 - 1

        self.fc = nn.Linear(in_features=latent_dims, out_features=self.c * 3)
        self.conv2 = nn.ConvTranspose1d(in_channels=self.c, out_channels=1, kernel_size=self.scalable_dim, stride=3, padding=1)

    def forward(self, x):
        x = self.fc(x)
        # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = x.view(x.size(0), self.c, 3)
        # x = F.relu(self.conv2(x))
        # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        x = torch.sigmoid(self.conv2(x))
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, batch_size, latent_dims, test_data):
        super(VariationalAutoencoder, self, ).__init__()

        self.latent_dims = latent_dims
        self.batch_size = batch_size

        temp_encooder = Encoder(self.batch_size, 1, 1)
        encoder_scalable_dim = temp_encooder.conv2(temp_encooder.conv1(test_data)).shape[2] * 2
        decoder_scalable_dim = test_data.shape[2]

        self.encoder = Encoder(self.batch_size, self.latent_dims, encoder_scalable_dim)
        self.decoder = Decoder(self.batch_size, self.latent_dims, decoder_scalable_dim)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def transform(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        return latent


def vae_loss(recon_x, x, mu, logvar, variational_beta=1):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    assert np.prod(x.shape) == np.prod(recon_x.shape), 'dimension error the shape mismatched %s and %s' % (str(x.shape),
                                                                                                           str(
                                                                                                               recon_x.shape))
    shape = np.prod(x.shape)
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, shape), x.view(-1, shape), reduction='sum')

    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence


def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
    x = [i + 1 for i in range(epoch + 1)]
    y = np.concatenate((train_loss, valid_loss))
    graphs = [[x, train_loss], [x, valid_loss]]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1 - x_margin, epochs + x_margin]
    y_bounds = [np.min(y) - y_margin, np.max(y) + y_margin]
    mb.update_graph(graphs, x_bounds, y_bounds)


def train_AE(num_epochs, vae, loader_train, loader_test, optimizer, device, verbose=False, save_dir=None):
    vae.train()

    mb = master_bar(range(num_epochs))
    best_val_loss = np.inf
    best_model_ = None

    mb.names = ['train', 'test']

    print('Training ...')

    train_loss_values, val_loss_values = [], []

    for epoch in mb:

        vae.train()

        train_loss_pre_epoch = list()
        for X, _ in progress_bar(loader_train, parent=mb):
            X = X.to(device)

            # vae reconstruction
            Z, latent_mu, latent_log_var = vae(X)
            loss = vae_loss(Z, X, latent_mu, latent_log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_pre_epoch.append(loss.item())

        train_loss_mean = np.mean(train_loss_pre_epoch)
        train_loss_values.append(train_loss_mean)

        vae.eval()
        val_loss_pre_epoch = list()
        for X, _ in progress_bar(loader_test, parent=mb):
            Z, latent_mu, latent_log_var = vae(X)
            loss = vae_loss(Z, X, latent_mu, latent_log_var)

            val_loss_pre_epoch.append(loss.item())
        val_loss_mean = np.mean(val_loss_pre_epoch)

        val_loss_values.append(val_loss_mean)

        if best_val_loss >= val_loss_mean:
            best_val_loss = val_loss_mean
            best_model_ = vae
            if save_dir:
                torch.save(vae.state_dict(), save_dir)

        if verbose:
            mb.main_bar.comment = f'EPOCHS'
            plot_loss_update(epoch, num_epochs, mb, train_loss_values, val_loss_values)

    return best_model_


if __name__ == '__main__':
    device = torch.device("cuda:3")

    X_train_transformed_path = './../../TDA-Datasets/Beef/Beef_TRAIN'
    X_test_transformed_path = './../../TDA-Datasets/Beef/Beef_TEST'

    # load the data
    X_train_transformed = np.load(X_train_transformed_path + '.npy')
    X_test_transformed = np.load(X_test_transformed_path + '.npy')

    print('X_train_transformed shape: ', X_train_transformed.shape)
    print('X_test_transformed shape:  ', X_test_transformed.shape)

    handle_dim = lambda x: np.swapaxes(x[..., np.newaxis], 1, -1)

    X_train_transformed_dim = TimeSeriesDataset(handle_dim(X_train_transformed), np.zeros(X_train_transformed.shape[0]))
    X_test_transformed_dim = TimeSeriesDataset(handle_dim(X_test_transformed), np.zeros(X_test_transformed.shape[0]))

    loader_train = DataLoader(X_train_transformed_dim, batch_size=32)
    loader_test = DataLoader(X_test_transformed_dim, batch_size=32)

    # ================================================================

    test_data = torch.zeros(X_train_transformed_dim[:][0].shape)

    vae = VariationalAutoencoder(32, 10, test_data)
    vae = vae.to(device)

    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=2e-3, weight_decay=1e-5)

    train_AE(10, vae, loader_train, loader_test, optimizer, device, verbose=True)
