import os
from random import randint

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
from fastprogress import progress_bar, master_bar
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset, DataLoader

from src.ae.vcae import vae_loss

DATA_PATH = './Univariate_arff/'


def readucr(filename):
    file = open(DATA_PATH + filename, 'rb')
    rows = [row for row in file]
    data = [row.split() for row in rows]
    data = np.array(data, dtype=np.float64)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def get_files_directory_list(path=None):
    if path is None:
        path = DATA_PATH

    directory_list = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            directory_list.append(name)
    return directory_list


def get_data_from_directory(fname, split=True):
    train_file_path = fname + '/' + fname + '_TRAIN' + '.txt'
    test_file_path = fname + '/' + fname + '_TEST' + '.txt'

    if os.path.isfile(train_file_path):
        raise FileNotFoundError('can\'t find the train file in this path %s' % train_file_path)
    if os.path.isfile(test_file_path):
        raise FileNotFoundError('can\'t find the test file in this path %s' % test_file_path)

    x_train, y_train = readucr(train_file_path)
    x_test, y_test = readucr(test_file_path)

    x_train, x_test, y_train, y_test = x_train[..., np.newaxis], x_test[..., np.newaxis], y_train[..., np.newaxis], \
                                       y_test[..., np.newaxis]

    if split:
        return x_train, x_test, y_train, y_test

    else:
        # concatenate the training and testing set
        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        return x, y


def one_hot_encoding(x):
    return pd.get_dummies(x).values

get_device = lambda: torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
handle_dim = lambda x, scale: np.swapaxes(scale.transform(x)[..., np.newaxis], 1, -1)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, device=None,):
        super(TimeSeriesDataset, self).__init__()

        self.device = get_device()

        self.X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(y, dtype=torch.float32).to(self.device)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TimeSeriesDataLoader(DataLoader):
    def __init__(self, X, y, batch_size=128, device=None,):
        time_series_dataset = TimeSeriesDataset(X, y, device)
        super(TimeSeriesDataLoader, self).__init__(time_series_dataset, batch_size=batch_size)



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


def train_clf(num_epochs, clf, loader_train, loader_test, optimizer, loss_fun, device, verbose=False, save_dir=None):
    clf.train()

    mb = master_bar(range(num_epochs))
    best_val_loss = np.inf
    best_model_ = None

    mb.names = ['train', 'test']

    print('Training ...')

    train_loss_values, val_loss_values = [], []

    for epoch in mb:

        clf.train()

        train_loss_pre_epoch = list()
        for X,y in progress_bar(loader_train, parent=mb):
            X = X.to(device)
            y = y.to(device)

            output = clf(X)
            loss = loss_fun(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_pre_epoch.append(loss.item())

        train_loss_mean = np.mean(train_loss_pre_epoch)
        train_loss_values.append(train_loss_mean)

        clf.eval()
        val_loss_pre_epoch = list()
        for X, y in progress_bar(loader_test, parent=mb):
            output = clf(X)
            loss = loss_fun(output, y)

            val_loss_pre_epoch.append(loss.item())
        val_loss_mean = np.mean(val_loss_pre_epoch)

        val_loss_values.append(val_loss_mean)

        if best_val_loss >= val_loss_mean:
            best_val_loss = val_loss_mean
            best_model_ = clf
            if save_dir:
                torch.save(clf.state_dict(), save_dir)

        if verbose:
            mb.main_bar.comment = f'EPOCHS'
            plot_loss_update(epoch, num_epochs, mb, train_loss_values, val_loss_values)

    return best_model_


def plot_clustering(z_run, labels, engine='plotly', download=False, folder_name='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """

    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        trace = go.Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = go.Data([trace])
        layout = go.Layout(
            title='PCA on z_run',
            showlegend=False
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = go.Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = go.Data([trace])
        layout = go.Layout(
            title='tSNE on z_run',
            showlegend=False
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download, folder_name):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='o', linewidths=0)
        plt.title('PCA on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "./pca.png")
        else:
            plt.show()

        plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='o', linewidths=0)
        plt.title('tSNE on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "./tsne.png")
        else:
            plt.show()

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)
