import spopt
from sklearn.metrics import pairwise as skm
import libpysal
from geo_utils import assign_hex_ids
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import concurrent.futures
import pandas as pd
import esda

def calculate_bootstrap_cluster_iteration(data, tessellate, aggregate, build_model, n_bootstraps, n_clusters, floor, distance_metric, over):
    result = []
    for iteration in range(n_bootstraps):
        sample = data.sample(frac=1, replace=True)
        sample, tessellation = tessellate(sample)
        tessellation = aggregate(sample, over, tessellation)
        weights = libpysal.weights.Queen.from_dataframe(tessellation, use_index=False)
        model = build_model(distance_metric, tessellation, weights, n_clusters, floor)
        model.solve()
        tessellation['cluster'] = model.labels_
        tessellation['n_clusters'] = n_clusters
        tessellation['floor'] = floor
        tessellation['distance_metric'] = distance_metric
        tessellation['algorithm'] = 'SKATER'
        tessellation['bootstrap_id'] = iteration
        result.append(tessellation)
    return result

class SpatialClustering:
    """
        Base class for spatial clustering 
    """
    def __init__(self, data, resolution, aggregates, metric, use_hexagons, n_bootstraps=1000):
        self.data = data
        self.metric = metric
        self.resolution = resolution
        self.aggregates = aggregates
        self.use_hexagons = use_hexagons
        self.n_bootstraps = n_bootstraps

    def aggregate(self, sample, over, tessellation):
        grouped = sample.groupby(over).agg(
            min=(self.metric, 'min'),
            max=(self.metric, 'max'),
            mean=(self.metric, 'mean'),
            std=(self.metric, lambda x: np.std(x) if len(x) > 1 else 0),
            p10=(self.metric, lambda x: np.percentile(x, 10)),
            p25=(self.metric, lambda x: np.percentile(x, 25)),
            p50=(self.metric, lambda x: np.percentile(x, 50)),
            p75=(self.metric, lambda x: np.percentile(x, 75)),
            p90=(self.metric, lambda x: np.percentile(x, 90)),
            p95=(self.metric, lambda x: np.percentile(x, 95)),
            p975 = (self.metric, lambda x: np.percentile(x, 97.5)),
            p99=(self.metric, lambda x: np.percentile(x, 99)),
            latency_reduction=(self.metric, lambda x: (np.percentile(x, 90) - np.percentile(x, 10))),
            norm_latency_reduction=(self.metric, lambda x: (np.percentile(x, 90) - np.percentile(x, 10)) / np.percentile(x, 10)),
            inequality_ratio = (self.metric, lambda x: np.percentile(x, 90) / np.percentile(x, 10))
        ).reset_index()
        geom = tessellation[[over, 'geometry']].drop_duplicates()
        geometries = {hid: geom for hid, geom in zip(geom[over], geom['geometry'])}
        geom_df = gpd.GeoDataFrame(grouped, geometry=[geometries[hid] for hid in grouped[over]])
        return geom_df

    @classmethod
    def calculate_jaccard_index(cls, a, b):
        """
            Calculate the Jaccard index
        """
        if len(a.union(b)) == 0:
            return 0
        return len(a.intersection(b)) / len(a.union(b))

    def aggregate_over_tessellation(self, over='hex_id'):
        """
            Tesselate the data and calculate aggregates
        """
        self.tessellation = self.aggregate(self.data, over, self.tessellation)

    def tessellate(self, sample):
        """
            Calculate the weights
        """
        if self.use_hexagons:
            sample, tessellation = assign_hex_ids(sample, self.resolution)  # Use existing geometries if we do not want to use hexagons
        else:
            tessellation = sample.copy()
        if tessellation.crs is None:
            tessellation = tessellation.set_crs("EPSG:4326")
        else:
            tessellation = tessellation.to_crs("EPSG:4326")
        return sample, tessellation

class SKATERClustering(SpatialClustering):
    """
        SKATER clustering algorithm
    """

    def __init__(self, data, resolution, aggregates, metric, use_hexagons, n_bootstraps=100):
        super().__init__(data, resolution, aggregates, metric, use_hexagons, n_bootstraps)
        self.metric_func = {
            'Manhattan': skm.manhattan_distances,
            'Euclidean': skm.euclidean_distances,
            'Chebyshev': lambda x, y: skm.pairwise_distances(x, y, metric='chebyshev'),
            'Cosine': skm.cosine_distances,
            'Correlation': lambda x, y: skm.pairwise_distances(x, y, metric='correlation'),
            'Canberra': lambda x, y: skm.pairwise_distances(x, y, metric='canberra'),
        }

    def build_model(self, distance_metric, tessellation, weights, n_clusters, floor):
        """
            Build the SKATER model
        """
        spanning_forest_kwds = dict(
            dissimilarity=self.metric_func[distance_metric],
            affinity=None,
            reduction=np.sum,
            center=np.mean,
        )
        model = spopt.region.Skater(
            tessellation,
            weights,
            self.aggregates,
            n_clusters=n_clusters,
            floor=floor,
            trace=False,
            islands="ignore",
            spanning_forest_kwds=spanning_forest_kwds
        )
        return model

    def cluster(self, n_clusters, floor, distance_metric, over='hex_id'):
        """
            Cluster the data
        """
        data, tessellation = self.tessellate(self.data)
        tessellation = self.aggregate(data, over, tessellation)
        self.weights = libpysal.weights.Queen.from_dataframe(tessellation, use_index=False)
        model = self.build_model(
            distance_metric,
            tessellation, 
            self.weights, 
            n_clusters,
            floor
        )
        model.solve()
        tessellation['cluster'] = model.labels_
        tessellation['n_clusters'] = n_clusters
        tessellation['floor'] = floor
        tessellation['distance_metric'] = distance_metric
        tessellation['algorithm'] = 'SKATER'
        return tessellation

    def get_optimal_clusters(self, distance_metric, over='hex_id'):
        """
            Get the optimal number of clusters and floor.

            Parameters:
                - floor: The floor value
                - distance_metric: The distance metric
                - over: The column to aggregate over

            Returns:
                - The optimal number of clusters and floor
        """
        data, tessellation = self.tessellate(self.data)
        tessellation = self.aggregate(data, over, tessellation)
        num_units = tessellation[over].nunique()
        floor = num_units // 10 # 10 is picked to ensure a decent runtime. It also ensures that the clusters are large enough to be meaningful.
        C0 = self.cluster(1, floor, distance_metric, over)
        prev = C0['cluster'].values
        K = 2
        while K <= 10:
            C1 = self.cluster(K, floor, distance_metric, over)
            curr = C1['cluster'].values
            if np.array_equal(prev, curr):
                return K-1, floor
            prev = curr
            K += 1
        return K, floor

    def plot(self, clus):
        """
            Plot the spatial clusters
        """
        fig, ax = plt.subplots(1, 1)
        clus['cluster'] = clus['cluster'].astype(str)
        clus.plot(column='cluster', cmap='tab20', legend=False, edgecolor='black', linewidth=0.5, ax=ax)
        plt.axis('off')
        # plt.savefig(fname, dpi=300, bbox_inches='tight')

    def calculate_bootstrap_cluster_iteration(self, iteration, n_clusters, floor, distance_metric, over):
        sample = self.data.sample(frac=1, replace=True)
        sample, tessellation = self.tessellate(sample)
        tessellation = self.aggregate(sample, over, tessellation)
        weights = libpysal.weights.Queen.from_dataframe(tessellation, use_index=False)
        model = self.build_model(distance_metric, tessellation, weights, n_clusters, floor)
        model.solve()
        tessellation['cluster'] = model.labels_
        tessellation['n_clusters'] = n_clusters
        tessellation['floor'] = floor
        tessellation['distance_metric'] = distance_metric
        tessellation['algorithm'] = 'SKATER'
        tessellation['bootstrap_id'] = iteration
        return tessellation

    def calculate_bootstrap_clusters(self, n_clusters, floor, distance_metric, over='hex_id'):
        """
            Calculate clusters for bootstrap samples.
        """
        data = []
        if self.data.crs is None:
            self.data.set_crs("EPSG:4326")
        else:
            self.data = self.data.to_crs(crs="EPSG:4326")
        for iteration in tqdm(range(self.n_bootstraps), desc="Bootstrapping..."):
            sample = self.data.sample(frac=1, replace=True)
            sample, tessellation = self.tessellate(sample)
            tessellation = self.aggregate(sample, over, tessellation)
            weights = libpysal.weights.Queen.from_dataframe(tessellation, use_index=False)
            model = self.build_model(distance_metric, tessellation, weights, n_clusters, floor)
            model.solve()
            tessellation['cluster'] = model.labels_
            tessellation['n_clusters'] = n_clusters
            tessellation['floor'] = floor
            tessellation['distance_metric'] = distance_metric
            tessellation['algorithm'] = 'SKATER'
            tessellation['bootstrap_id'] = iteration
            data.append(tessellation)
        return pd.concat(data, axis=0)

    def calculate_jaccard_matrix(self, tessellation, n_clusters, over='hex_id'):
        """
            Calculate Jaccard matrix from bootstrapped clusters.
        """
        jaccard_matrix = np.zeros((n_clusters, self.n_bootstraps, self.n_bootstraps))
        for cid in range(n_clusters):
            for i in range(self.n_bootstraps):
                for j in range(i, self.n_bootstraps):
                    a = set(tessellation[(tessellation['bootstrap_id'] == i) & (tessellation['cluster'] == cid)][over].values)
                    b = set(tessellation[(tessellation['bootstrap_id'] == j) & (tessellation['cluster'] == cid)][over].values)
                    jaccard_matrix[cid, i, j] = self.calculate_jaccard_index(a, b)
                    jaccard_matrix[cid, j, i] = jaccard_matrix[cid, i, j]
        return jaccard_matrix

    def plot_jaccard_matrix(self, jaccard_matrix):
        """
            Plot the Jaccard matrix.
        """
        plt.imshow(jaccard_matrix, cmap='viridis')
        plt.colorbar()
        plt.show()
