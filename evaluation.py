from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import accuracy_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
### our imports

from src.utils import (get_data_from_directory, get_files_directory_list, 
                       one_hot_encoding, TimeSeriesDataset,get_device, train_clf, train_AE)

from src.TFE import *
from src import VariationalAutoencoder

from pathlib import Path
import json


def evaluate_dataset(dataset_path):
    print("Evaluating " + str(dataset_path))
    dataset_name = dataset_path.stem
    X_train_transformed = np.load(dataset_path / (dataset_name + "_TRAIN.npy"))
    X_test_transformed = np.load(dataset_path / (dataset_name + "_TEST.npy"))

    directory_list = get_files_directory_list()
    directory_list = sorted(directory_list)

    dataset_index = 0
    for i, name in enumerate(directory_list):
        if name == dataset_name:
            dataset_index = i
            break

    _1, _2, y_train, y_test = get_data_from_directory(directory_list[dataset_index])
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Without VAE
    results = dict()

    print("Training SVM, no AE...")
    parameters = {"C": [10**i for i in range(-2, 5)],
              "kernel": ["linear", "rbf", "sigmoid", "poly"]}
    svc_cv = GridSearchCV(SVC(random_state=42), 
                        param_grid=parameters,
                        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                        scoring='accuracy', 
                        n_jobs=-1)
    svc_cv.fit(X_train_transformed, y_train)
    acc_train = accuracy_score(y_train, svc_cv.best_estimator_.predict(X_train_transformed))
    acc_test = accuracy_score(y_test, svc_cv.best_estimator_.predict(X_test_transformed))
    print("SVM, no AE, acc_train: " + str(acc_train) + ", acc_test: " + str(acc_test))
    results["svm_no_ae"] = {"accuracy": (acc_train, acc_test), "params": svc_cv.best_params_}

    print("Training XGBoost, no AE...")
    parameters = {"max_depth": [2, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 120, 150],
              "n_estimators": [20, 50, 100, 150, 200, 250]}
    xgb_cv = GridSearchCV(XGBClassifier(n_jobs=-1, random_state=42), 
                          param_grid=parameters,
                          cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                          scoring='accuracy', 
                          n_jobs=-1)
    xgb_cv.fit(X_train_transformed, y_train)
    acc_train = accuracy_score(y_train, xgb_cv.best_estimator_.predict(X_train_transformed))
    acc_test = accuracy_score(y_test, xgb_cv.best_estimator_.predict(X_test_transformed))
    print("XGBoost, no AE, acc_train: " + str(acc_train) + ", acc_test: " + str(acc_test))
    results["xgboost_no_ae"] = {"accuracy": (acc_train, acc_test), "params": xgb_cv.best_params_}

    print("Training KNN, no AE...")
    parameters = {"n_neighbors": [3, 5, 7, 11,]}
    knn_cv = GridSearchCV(KNeighborsClassifier(n_jobs=-1), 
                        param_grid=parameters,
                        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                        scoring='accuracy', 
                        n_jobs=-1)
    knn_cv.fit(X_train_transformed, y_train)
    acc_train = accuracy_score(y_train, knn_cv.best_estimator_.predict(X_train_transformed))
    acc_test = accuracy_score(y_test, knn_cv.best_estimator_.predict(X_test_transformed))
    print("KNN, no AE, acc_train: " + str(acc_train) + ", acc_test: " + str(acc_test))
    results["knn_no_ae"] = {"accuracy": (acc_train, acc_test), "params": knn_cv.best_params_}


    print("Training VAE...")

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    num_classes = np.unique(y_train).shape[0]
    device = get_device()
    batch_size = 32
    latent_dim = num_classes * 4

    scale = StandardScaler()
    scale.fit(X_train_transformed)

    handle_dim = lambda x: np.swapaxes(scale.transform(x)[..., np.newaxis], 1, -1)

        
    X_train_transformed_dim = handle_dim(X_train_transformed)
    X_test_transformed_dim  = handle_dim(X_test_transformed)

    y_hot_train = one_hot_encoding(y_train)
    y_hot_test = one_hot_encoding(y_test)

    dataset_train = TimeSeriesDataset(X_train_transformed_dim, y_hot_train)
    dataset_test  = TimeSeriesDataset(X_test_transformed_dim, y_hot_test)

    loader_train = DataLoader(dataset_train, batch_size=batch_size)
    loader_test = DataLoader(dataset_test, batch_size=batch_size)

    test_data = torch.zeros(dataset_train[:][0].shape)

    vae = VariationalAutoencoder(batch_size=batch_size, latent_dims=latent_dim, test_data=test_data)
    vae = vae.to(device)

    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=2e-3, weight_decay=1e-5)

    train_AE(1000, vae, loader_train, loader_test, optimizer, device, verbose=True)

    from_vae_loader2numpy = lambda model, x: model.transform(x.dataset[:][0]).cpu().detach().numpy()
    X_train_transformed = from_vae_loader2numpy(vae, loader_train)
    X_test_transformed = from_vae_loader2numpy(vae, loader_test)

    print("Training SVM, with AE...")
    parameters = {"C": [10**i for i in range(-2, 5)],
              "kernel": ["linear", "rbf", "sigmoid", "poly"]}
    svc_cv = GridSearchCV(SVC(random_state=42), 
                        param_grid=parameters,
                        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                        scoring='accuracy', 
                        n_jobs=-1)
    svc_cv.fit(X_train_transformed, y_train)
    acc_train = accuracy_score(y_train, svc_cv.best_estimator_.predict(X_train_transformed))
    acc_test = accuracy_score(y_test, svc_cv.best_estimator_.predict(X_test_transformed))
    print("SVM, with AE, acc_train: " + str(acc_train) + ", acc_test: " + str(acc_test))
    results["svm_with_ae"] = {"accuracy": (acc_train, acc_test), "params": svc_cv.best_params_}

    print("Training XGBoost, with AE...")
    parameters = {"max_depth": [2, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 120, 150],
              "n_estimators": [20, 50, 100, 150, 200, 250]}
    xgb_cv = GridSearchCV(XGBClassifier(n_jobs=-1, random_state=42), 
                          param_grid=parameters,
                          cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                          scoring='accuracy', 
                          n_jobs=-1)
    xgb_cv.fit(X_train_transformed, y_train)
    acc_train = accuracy_score(y_train, xgb_cv.best_estimator_.predict(X_train_transformed))
    acc_test = accuracy_score(y_test, xgb_cv.best_estimator_.predict(X_test_transformed))
    print("XGBoost, with AE, acc_train: " + str(acc_train) + ", acc_test: " + str(acc_test))
    results["xgboost_with_ae"] = {"accuracy": (acc_train, acc_test), "params": xgb_cv.best_params_}

    print("Training KNN, with AE...")
    parameters = {"n_neighbors": [3, 5, 7, 11,]}
    knn_cv = GridSearchCV(KNeighborsClassifier(n_jobs=-1), 
                        param_grid=parameters,
                        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                        scoring='accuracy', 
                        n_jobs=-1)
    knn_cv.fit(X_train_transformed, y_train)
    acc_train = accuracy_score(y_train, knn_cv.best_estimator_.predict(X_train_transformed))
    acc_test = accuracy_score(y_test, knn_cv.best_estimator_.predict(X_test_transformed))
    print("KNN, with AE, acc_train: " + str(acc_train) + ", acc_test: " + str(acc_test))
    results["knn_with_ae"] = {"accuracy": (acc_train, acc_test), "params": knn_cv.best_params_}

    return results
    

def main():
    base_path = Path("./TDA-Datasets")
    print("Starting evaluation")
    total_results = list()
    for dataset_path in base_path.iterdir():
        try:
            results = evaluate_dataset(dataset_path)
            total_results.append(results)
        except Exception as e:
            print("Error: " + str(e))
    print("Evaluation finished")

    with open('evaluation.json', 'w') as outfile:
        json.dump(total_results, outfile)




if __name__ == "__main__":
    main()