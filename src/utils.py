import os
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

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

    x_train, x_test, y_train, y_test = x_train[..., np.newaxis], x_test[..., np.newaxis], y_train[..., np.newaxis], y_test[..., np.newaxis]

    if split:
        return x_train, x_test, y_train, y_test

    else:
        # concatenate the training and testing set
        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        return x, y


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


def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]
