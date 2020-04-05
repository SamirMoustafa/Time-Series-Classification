import torch
from torch import nn

from src.utils import train_clf, TimeSeriesDataLoader, one_hot_encoding, inverse_one_hot_encoding


class ANN(object):
    def __init__(self, latent_dim, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        self.clf = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=latent_dim * 2),
                                 nn.BatchNorm1d(latent_dim * 2),
                                 nn.Dropout(.3),

                                 nn.Linear(in_features=latent_dim * 2, out_features=latent_dim * 3),
                                 nn.BatchNorm1d(latent_dim * 3),
                                 nn.Dropout(.3),

                                 nn.Linear(in_features=latent_dim * 3, out_features=latent_dim // 2),
                                 nn.BatchNorm1d(latent_dim // 2),
                                 nn.Dropout(.3),

                                 nn.Linear(in_features=latent_dim // 2, out_features=num_classes),
                                 nn.Sigmoid()).to(device)

        self.from_clf_loader2numpy = lambda model, x: model(x.dataset[:][0]).cpu().detach().numpy()

    def fit(self, X, y):
        y = one_hot_encoding(y)
        dataset_train = TimeSeriesDataLoader(X, y, 128)
        optimizer = torch.optim.SGD(params=self.clf.parameters(), lr=1e-3, momentum=.9)
        loss_fun = nn.BCELoss()
        train_clf(50, self.clf, dataset_train, [], optimizer, loss_fun, self.device, True)
        return self

    def predict(self, X):
        _y = torch.zeros(X.shape[0], self.num_classes)
        dataset_train = TimeSeriesDataLoader(X, _y, 128)
        y = inverse_one_hot_encoding(self.from_clf_loader2numpy(self.clf, dataset_train))
        return y