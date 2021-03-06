from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, PersistenceLandscape, BettiCurve
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding

from src.TFE.base import PersistenceDiagramFeatureExtractor


class PersistenceDiagramsExtractor:
    def __init__(self, tokens_embedding_dim, tokens_embedding_delay, homology_dimensions,
                 filtering=False, filtering_dimensions=(1, 2), parallel=False):
        self.tokens_embedding_dim_ = tokens_embedding_dim
        self.tokens_embedding_delay_ = tokens_embedding_delay
        self.homology_dimensions_ = homology_dimensions
        self.filtering_ = filtering
        self.filtering_dimensions_ = filtering_dimensions
        self.parallel_ = parallel
        self.n_job = -1 if self.parallel_ else None

    def tokens_embeddings_(self, X):
        X_transformed = list()
        for series in X:
            te = TakensEmbedding(parameters_type='search',
                                 dimension=self.tokens_embedding_dim_,
                                 time_delay=self.tokens_embedding_delay_,
                                 n_jobs=self.n_job)
            X_transformed.append(te.fit_transform(series))
        return X_transformed

    def persistence_diagrams_(self, X_embdeddings):
        if self.parallel_:
            pool = ThreadPool()
            X_transformed = pool.map(self.parallel_embed_, X_embdeddings)
            pool.close()
            pool.join()
            return X_transformed
        else:
            X_transformed = list()
            for embedding in X_embdeddings:
                X_transformed.append(self.parallel_embed_(embedding))
            return X_transformed

    def parallel_embed_(self, embedding):
        vr = VietorisRipsPersistence(metric='euclidean', homology_dimensions=self.homology_dimensions_,
                                     n_jobs=self.n_job)
        diagram_scaler = Scaler(n_jobs=self.n_job)
        persistence_diagrams = diagram_scaler.fit_transform(vr.fit_transform([embedding]))
        if self.filtering_:
            diagram_filter = Filtering(epsilon=0.1, homology_dimensions=self.filtering_dimensions_)
            persistence_diagrams = diagram_filter.fit_transform(persistence_diagrams)
        return persistence_diagrams[0]

    def fit_transform(self, X):
        X_embeddings = self.tokens_embeddings_(X)
        X_persistence_diagrams = self.persistence_diagrams_(X_embeddings)
        return X_persistence_diagrams


class TopologicalFeaturesExtractor:
    def __init__(self, persistence_diagram_extractor, persistence_diagram_features):
        self.persistence_diagram_extractor_ = persistence_diagram_extractor
        self.persistence_diagram_features_ = persistence_diagram_features

    def fit_transform(self, X):
        X_transformed = None

        X_pd = self.persistence_diagram_extractor_.fit_transform(X)

        for pd_feature in self.persistence_diagram_features_:
            X_features = pd_feature.fit_transform(X_pd)
            X_transformed = np.vstack((X_transformed.T, X_features.T)).T if X_transformed is not None else X_features

        return X_transformed


class HolesNumberFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(HolesNumberFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        # Persistence diagrams has dimensionality (N, 3).
        # The second coordinate has following notation: (birth, death, homology dimension).
        # See https://github.com/giotto-ai/giotto-tda/blob/master/gtda/plotting/persistence_diagrams.py
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        for hole in persistence_diagram:
            if hole[1] - hole[0] > 0:
                # Some holes still can have zero-lifetime
                # See https://giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/preprocessing/gtda.diagrams.Filtering.html#gtda.diagrams.Filtering.fit_transform
                feature[int(hole[2])] += 1.0
        return feature


class MaxHoleLifeTimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(MaxHoleLifeTimeFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        for hole in persistence_diagram:
            lifetime = hole[1] - hole[0]
            if lifetime > feature[int(hole[2])]:
                feature[int(hole[2])] = lifetime
        return feature


class RelevantHolesNumber(PersistenceDiagramFeatureExtractor):
    def __init__(self, ratio=0.7):
        super(RelevantHolesNumber).__init__()
        self.ratio_ = ratio

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        max_lifetimes = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)

        for hole in persistence_diagram:
            lifetime = hole[1] - hole[0]
            if lifetime > max_lifetimes[int(hole[2])]:
                max_lifetimes[int(hole[2])] = lifetime

        for hole in persistence_diagram:
            index = int(hole[2])
            lifetime = hole[1] - hole[0]
            if np.equal(lifetime, self.ratio_ * max_lifetimes[index]):
                feature[index] += 1.0

        return feature


class AverageHoleLifetimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(AverageHoleLifetimeFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        n_holes = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)

        for hole in persistence_diagram:
            lifetime = hole[1] - hole[0]
            index = int(hole[2])
            if lifetime > 0:
                feature[index] += lifetime
                n_holes[index] += 1

        for i in range(feature.shape[0]):
            feature[i] = feature[i] / n_holes[i] if n_holes[i] != 0 else 0.0

        return feature


class SumHoleLifetimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(SumHoleLifetimeFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        for hole in persistence_diagram:
            feature[int(hole[2])] += hole[1] - hole[0]
        return feature


class PersistenceEntropyFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(PersistenceEntropyFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        persistence_entropy = PersistenceEntropy(n_jobs=-1)
        return persistence_entropy.fit_transform([persistence_diagram])[0]


class SimultaneousAliveHolesFeatue(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(SimultaneousAliveHolesFeatue).__init__()

    def get_average_intersection_number_(self, segments):
        intersections = list()
        n_segments = segments.shape[0]
        i = 0

        for i in range(0, n_segments):
            count = 1
            start = segments[i, 0]
            end = segments[i, 1]

            for j in range(i + 1, n_segments):
                if segments[j, 0] >= start and segments[j, 0] <= end:
                    count += 1
                else:
                    break
            intersections.append(count)

        return np.sum(intersections) / len(intersections)

    def get_average_simultaneous_holes_(self, holes):
        starts = holes[:, 0]
        ends = holes[:, 1]
        ind = np.lexsort((starts, ends))
        segments = np.array([[starts[i], ends[i]] for i in ind])
        return self.get_average_intersection_number_(segments)

    def extract_feature_(self, persistence_diagram):
        n_dims = int(np.max(persistence_diagram[:, 2])) + 1
        feature = np.zeros(n_dims)

        for dim in range(n_dims):
            holes = list()
            for hole in persistence_diagram:
                if hole[1] - hole[0] != 0.0 and int(hole[2]) == dim:
                    holes.append(hole)
            if len(holes) != 0:
                feature[dim] = self.get_average_simultaneous_holes_(np.array(holes))

        return feature


class AveragePersistenceLandscapeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(AveragePersistenceLandscapeFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        # As practice shows, only 1st layer of 1st homology dimension plays role
        persistence_landscape = PersistenceLandscape(n_jobs=-1).fit_transform([persistence_diagram])[0, 1, 0, :]
        return np.array([np.sum(persistence_landscape) / persistence_landscape.shape[0]])


class BettiNumbersSumFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(BettiNumbersSumFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        betti_curve = BettiCurve(n_jobs=-1).fit_transform([persistence_diagram])[0]
        return np.array([np.sum(betti_curve[i, :]) for i in range(int(np.max(persistence_diagram[:, 2])) + 1)])


class RadiusAtMaxBNFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(RadiusAtMaxBNFeature).__init__()

    def extract_feature_(self, persistence_diagram, n_bins=100):
        betti_curve = BettiCurve(n_jobs=-1, n_bins=n_bins).fit_transform([persistence_diagram])[0]
        max_dim = int(np.max(persistence_diagram[:, 2])) + 1
        max_bettis = np.array([np.max(betti_curve[i, :]) for i in range(max_dim)])
        return np.array(
            [np.where(betti_curve[i, :] == max_bettis[i])[0][0] / (n_bins * max_dim) for i in range(max_dim)])
