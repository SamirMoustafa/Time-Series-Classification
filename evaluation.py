import warnings
from multiprocessing.dummy import Pool as ThreadPool

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from torch import nn

### our imports
from src.utils import *
from src.TFE import *
from src import VariationalAutoencoder

from pathlib import Path
import json

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
device = get_device()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_data_index_from_filename(file_name, directory_list):
    for i, name in enumerate(directory_list):
        if name == file_name:
            return i


def handle_n_neighbors_for_lower_dim_data(n_neighbors, shape):
    # a lot of explanation, call it without understanding ))
    n_neighbors = np.array(n_neighbors)
    n_neighbors = n_neighbors[np.where(n_neighbors <= shape[0]//2)]
    return n_neighbors


def run_single_model(model, params, X_train, X_test, y_train, y_test, is_vae):
    results = dict()
    clf_name = type(model).__name__
    with_ = 'with' if is_vae else 'without'
    clf_cv = GridSearchCV(model,
                          param_grid=params,
                          cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                          scoring='accuracy',
                          n_jobs=-1)

    clf_cv.fit(X_train, y_train)
    y_pred_train = clf_cv.best_estimator_.predict(X_train)
    y_pred_test = clf_cv.best_estimator_.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)


    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')


    precision_train = precision_score(y_train, y_pred_train, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')

    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    print(clf_name + ", " + with_ + " AE, acc_train: " + str(acc_train) + ", acc_test: " + str(acc_test))

    results[clf_name] = {"accuracy": (acc_train, acc_test),
                         "recall": (recall_train, recall_test),
                         "precision": (precision_train, precision_test),
                         "f1": (f1_train, f1_test),
                         "params": clf_cv.best_params_}
    return results


def run_models_list(dataset_name, models_list, params_list, X_train, X_test, y_train, y_test, is_vae):
    with_ = ('with' if is_vae else 'without') + '_AE'

    pool = ThreadPool()
    results_list = list()
    results_dict = dict()
    for i, clf in enumerate(models_list):
        parameters = params_list[i]

        if 'n_neighbors' in parameters.keys():
            parameters['n_neighbors'] = handle_n_neighbors_for_lower_dim_data(parameters['n_neighbors'], X_train.shape)

        results_list.append(pool.apply_async(run_single_model, (clf, parameters, X_train, X_test, y_train, y_test, is_vae)))

    pool.close()
    pool.join()

    res = [p.get() for p in results_list]
    [results_dict.update(res) for res in res]
    return {dataset_name+'_'+with_: results_dict}


def evaluate_dataset(dataset_path):
    print("Evaluating " + str(dataset_path))
    dataset_name = dataset_path.stem

    X_train_transformed = np.load(dataset_path / (dataset_name + "_TRAIN.npy"))
    X_test_transformed = np.load(dataset_path / (dataset_name + "_TEST.npy"))

    directory_list = get_files_directory_list()
    directory_list = sorted(directory_list)

    dataset_index = get_data_index_from_filename(dataset_name, directory_list)

    _, _, y_train, y_test = get_data_from_directory(directory_list[dataset_index])

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    models_list = [ #SVC(random_state=42),
                    #XGBClassifier(n_jobs=-1, random_state=42),
                    KNeighborsClassifier(n_jobs=-1),
                    #CatBoostClassifier(random_state=42, verbose=False, silent=True),
                    RandomForestClassifier(n_jobs=-1, random_state=4)]

    params_list = [#{"C": [10 ** i for i in range(-2, 1)],
                   # "kernel": ["linear", "rbf", "sigmoid", "poly"]},

                   #{"max_depth": [2, 35, 70, 120, 150],
                   # "n_estimators": [20, 50, 100, ]},

                   {"n_neighbors": [3, 5, 7, 11, ]},

                   #{"max_depth": [2, 35, 70, 120, 150],
                   # "n_estimators": [20, 50, 100,],
                   # "early_stopping_rounds": [2, 5, 8, 10, 50, 200]},

                   {"max_depth": [2, 35, 70, 120, 150],
                    "n_estimators": [20, 50, 100, ]}, ]

    results = run_models_list(dataset_name,
                              models_list,
                              params_list,
                              X_train_transformed, X_test_transformed,
                              y_train, y_test,
                              False)

    # Without VAE

    print("Training VAE...")

    num_classes = np.unique(y_train).shape[0]
    batch_size = 32
    latent_dim = 4

    scale = StandardScaler()
    scale.fit(X_train_transformed)

    X_train_transformed_dim = handle_dim(X_train_transformed, scale)
    X_test_transformed_dim = handle_dim(X_test_transformed, scale)

    y_hot_train = one_hot_encoding(y_train)
    y_hot_test = one_hot_encoding(y_test)

    dataset_train = TimeSeriesDataLoader(X_train_transformed_dim, y_hot_train, batch_size)
    dataset_test = TimeSeriesDataLoader(X_test_transformed_dim, y_hot_test, batch_size)

    test_data = torch.zeros(dataset_train.dataset[:][0].shape)

    vae = VariationalAutoencoder(batch_size=batch_size, latent_dims=latent_dim, test_data=test_data)
    vae = vae.to(device)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=2e-3, weight_decay=1e-5)

    vae = train_AE(1, vae, dataset_train, dataset_test, optimizer, device, verbose=True)

    from_vae_loader2numpy = lambda model, x: model.transform(x.dataset[:][0]).cpu().detach().numpy()
    z_train = from_vae_loader2numpy(vae, dataset_train)
    z_test = from_vae_loader2numpy(vae, dataset_test)

    results_with_vae = run_models_list(dataset_name,
                                       models_list,
                                       params_list,
                                       z_train, z_test,
                                       y_train, y_test,
                                       True)

    dataset_train = TimeSeriesDataLoader(z_train, y_hot_train, batch_size)
    dataset_test = TimeSeriesDataLoader(z_test, y_hot_test, batch_size)

    clf = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=latent_dim * 2),
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

    optimizer = torch.optim.SGD(params=clf.parameters(), lr=1e-3, momentum=.9)
    loss_fun = nn.BCELoss()

    train_clf(2, clf, dataset_train, dataset_test, optimizer, loss_fun, device, True)

    from_clf_loader2numpy = lambda model, x: model(x.dataset[:][0]).cpu().detach().numpy()

    y_train_pred = from_clf_loader2numpy(clf, dataset_train)
    y_test_pred = from_clf_loader2numpy(clf, dataset_test)

    y_train_pred_norm = np.zeros_like(y_train_pred)
    y_train_pred_norm[np.arange(len(y_train_pred)), y_train_pred.argmax(1)] = 1

    y_test_pred_norm = np.zeros_like(y_test_pred)
    y_test_pred_norm[np.arange(y_test_pred.shape[0]), y_test_pred.argmax(1)] = 1

    print("Train accuracy: ", )
    print("Test accuracy: ", accuracy_score(y_hot_test, y_test_pred_norm))

    acc_train = accuracy_score(y_hot_train, y_train_pred_norm)
    acc_test = accuracy_score(y_hot_test, y_test_pred_norm)

    print("ANN, with AE, acc_train: " + str(acc_train) + ", acc_test: " + str(acc_test))

    neural_net_params = {'in_features=latent_dim': latent_dim,
                         'out_features': num_classes,
                         'depth': 4}

    results_with_vae["NeuralNet_with_ae"] = {"accuracy": (acc_train, acc_test), "params": neural_net_params}
    results.update(results_with_vae)
    return results


def main():
    base_path = Path("./TDA-Datasets")
    print("Starting evaluation")
    total_results = list()
    for dataset_path in base_path.iterdir():
        # try:
        results = evaluate_dataset(dataset_path)
        total_results.append(results)
        # except Exception as e:
        #    print("Error: " + str(e))
    print("Evaluation finished")

    with open('evaluation.json', 'w') as outfile:
        json.dump(total_results, outfile, cls=NpEncoder)


if __name__ == "__main__":
    main()
