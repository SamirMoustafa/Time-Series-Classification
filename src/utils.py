import os
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset

DATA_PATH = './../../Univariate_arff/'


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


def one_hot_encoding(x, dtype=float):
    x = np.asarray(x).astype(int) - 1
    n = np.unique(x).shape[0]
    return np.eye(int(n), dtype=dtype)[x]


get_device = lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, device=None, ):
        super(TimeSeriesDataset, self).__init__()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.device = get_device()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].to(self.device), self.y[idx].to(self.device)


def train_AE(epochs, net, criterion, optimizer, train_loader, val_loader, scheduler=None, verbose=True, save_dir=None):
    net.to(get_device())
    for epoch in range(1, epochs + 1):
        net.train()
        for X, _ in train_loader:
            # Perform one step of minibatch stochastic gradient descent

            # >>> your solution here <<<
            optimizer.zero_grad()
            output = net(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()

        # define NN evaluation, i.e. turn off dropouts, batchnorms, etc.
        net.eval()
        best_val_loss = np.inf
        for X, _ in val_loader:
            # Compute the validation loss

            # >>> your solution here <<<
            output = net(X)
            val_loss = criterion(output, X)

            if best_val_loss >= val_loss.item() and save_dir:
                best_val_loss = val_loss.item()
                torch.save(net.state_dict(), save_dir)

        if scheduler is not None:
            scheduler.step()
        freq = max(epochs // 20, 1)
        if verbose and epoch % freq == 0:
            print('Epoch {}/{} ||\t Loss:  Train {:.4f} | Validation {:.4f}'.format(epoch,
                                                                                  epochs,
                                                                                  loss.item(),
                                                                                  val_loss.item()))


def train_clf(epochs, net, criterion, optimizer, train_loader, val_loader, scheduler=None, verbose=True, save_dir=None):
    net.to(get_device())
    for epoch in range(1, epochs + 1):
        net.train()
        for X, y in train_loader:
            # Perform one step of minibatch stochastic gradient descent

            # >>> your solution here <<<
            optimizer.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # define NN evaluation, i.e. turn off dropouts, batchnorms, etc.
        net.eval()
        best_val_loss = np.inf
        for X, y in val_loader:
            # Compute the validation loss

            # >>> your solution here <<<
            output = net(X)
            val_loss = criterion(output, y)

            if best_val_loss >= val_loss.item() and save_dir:
                best_val_loss = val_loss.item()
                torch.save(net.state_dict(), save_dir)

        if scheduler is not None:
            scheduler.step()
        freq = max(epochs // 20, 1)
        if verbose and epoch % freq == 0:
            print('Epoch {}/{} ||\t Loss:  Train {:.4f} | Validation {:.4f}'.format(epoch,
                                                                                  epochs,
                                                                                  loss.item(),
                                                                                  val_loss.item()))


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
