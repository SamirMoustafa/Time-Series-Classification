# Before running the script, run:
# !wget -nc "http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip"
# !unzip -q -n "Univariate2018_arff.zip"

import numpy as np
import pandas as pd

import sys
from pathlib import Path

from src.utils import (get_data_from_directory, get_files_directory_list, 
                       one_hot_encoding, TimeSeriesDataset,get_device, train_clf)

from src.TFE import *


def extract_dataset(dataset_index):
    directory_list = get_files_directory_list()
    directory_list = sorted(directory_list)

    dataset_name = directory_list[dataset_index]

    print("Processing dataset " + str(dataset_index) + ": " + dataset_name + "...")

    X_train, X_test, y_train, y_test = get_data_from_directory(dataset_name)
    X_train = X_train.squeeze()
    y_train = y_train.squeeze()
    X_test = X_test.squeeze()
    y_test = y_test.squeeze()

    feature_extractor = TopologicalFeaturesExtractor(
        persistence_diagram_extractor=PersistenceDiagramsExtractor(tokens_embedding_dim=2, 
                                                                   tokens_embedding_delay=3,
                                                                   homology_dimensions=(0, 1),
                                                                   parallel=True),
        persistence_diagram_features=[HolesNumberFeature(),
                                      MaxHoleLifeTimeFeature(),
                                      RelevantHolesNumber(),
                                      AverageHoleLifetimeFeature(),
                                      SumHoleLifetimeFeature(),
                                      PersistenceEntropyFeature(),
                                      SimultaneousAliveHolesFeatue(),
                                      AveragePersistenceLandscapeFeature(),
                                      BettiNumbersSumFeature(),
                                      RadiusAtMaxBNFeature()])

    X_train_transformed = feature_extractor.fit_transform(X_train)
    X_test_transformed = feature_extractor.fit_transform(X_test)

    base_path = Path("./TDA-Datasets/")
    if not base_path.exists():
        base_path.mkdir()
    dataset_path = base_path / dataset_name
    dataset_path.mkdir()

    np.save(dataset_path / (dataset_name + "_TRAIN"), X_train_transformed)
    np.save(dataset_path / (dataset_name + "_TEST"), X_test_transformed)

    print("Dataset " + str(dataset_index) + " finished")


def main():
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])
    for i in range(start_index, end_index + 1):
        try:
            extract_dataset(i)
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    main()