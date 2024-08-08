import numpy as np
from h3 import h3
import statsmodels.api as sm
import pandas as pd
from loess.loess_2d import loess_2d
import geopandas as gpd
from shapely.geometry import Polygon, Point
from joblib import Parallel, delayed
from tqdm import tqdm
from data_utils import *
from sklearn.neighbors import KernelDensity, NearestNeighbors
from geo_utils import *
from multiprocessing import Pool
import time
from sklearn.ensemble import RandomForestRegressor
from pykrige.rk import RegressionKriging
from pykrige.ok import OrdinaryKriging

class Smoother:
    """
    A parent class for all smoothers.

    Attributes:
        data (GeoDataFrame): The input data.
        params (dict): Dictionary of parameters.
        coord_columns (list): List of column names for coordinates.
        h3_resolution (int): H3 resolution level.
        agg_fns (dict): Dictionary of aggregation functions.
        grid_spacing (float): Spacing between grid points.
        metric (str): Name of the metric to smooth.
        epsg (int): EPSG code for the coordinate reference system.
        tessellation (GeoDataFrame): Tessellation of the input data.
        test_data (GeoDataFrame): Test data for validation.
        shapefile (str): Path to the shapefile if testing on a specific region without considering hexagons.
    """
    def __init__(self, data, params, shapefile=None, test_data=None, no_grid=True):
        self.params = params
        self.coord_columns = self.params['coord_columns']
        self.h3_resolution = self.params['h3_resolution']
        self.agg_fns = self.params['agg_fns']
        self.grid_spacing = self.params['grid_spacing']
        self.metric = self.params['metric']
        self.epsg = self.params['epsg']
        lng, lat = data[self.coord_columns[0]], data[self.coord_columns[1]]
        self.data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(lng, lat), crs='EPSG:4326')
        self.tessellation = None
        self.test_data = test_data
        if self.test_data is not None:
            self.test_data = gpd.GeoDataFrame(
                test_data, 
                geometry=gpd.points_from_xy(
                    test_data[self.coord_columns[0]], 
                    test_data[self.coord_columns[1]]
                ), 
                crs='EPSG:4326'
            )
            self.test_data = self.test_data.to_crs(self.epsg)
        self.shapefile = shapefile
        self.no_grid = no_grid

    def fit(self, n_jobs=-1):
        """
        Fits a smoothing model to the data.

        Returns:
            sdata: A data frame containing smoothed values for each cell along with weights to filter outliers.
        """
        self.data , self.tessellation = assign_hex_ids(self.data, self.h3_resolution)
        data = {'h3_cell': [], 'z_smooth': [], 'w_smooth': [], 'grid_x': [], 'grid_y': []}
        data = {**data, **{k: [] for k in self.agg_fns.keys()}}

        def process_cell(cell):
            cell_data = self.data[self.data['hex_id'] == cell]
            grid_x, grid_y, grid_z, z_smooth, w_smooth, coeff = self.smooth(cell)
            sample_size = len(cell_data)
            if z_smooth is None:
                agg_results = {k: np.nan for k, v in self.agg_fns.items()}
                z_smooth = [] # To filter these samples out later on
                w_smooth = []
            else:
                agg_results = {k: v(z_smooth) for k, v in self.agg_fns.items()}
                z_smooth = list(z_smooth)
                w_smooth = list(w_smooth)
            return {'h3_cell': cell, 'sample_size': sample_size, 'z_smooth': z_smooth, 
            'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z, 'coeff': coeff,
            'w_smooth': w_smooth, **agg_results}

        # TODO: This is hacky. Flag it and think about design later.
        if self.shapefile is not None or self.no_grid:
            grid_x, grid_y, z_ground, z_smooth, w_smooth = self.smooth_regionwide()
            return pd.DataFrame({'grid_x': grid_x, 'grid_y': grid_y, 'z_ground': z_ground, 'z_smooth': z_smooth, 'w_smooth': w_smooth})
        else:
            cells = list(self.data['hex_id'].unique())
            results = Parallel(n_jobs=n_jobs)(delayed(process_cell)(cell) for cell in tqdm(cells, desc='Smoothing', leave=False))
            sdata = pd.DataFrame(results)
            return sdata

class AdaptiveKDESmoother(Smoother):
    """
    A class to perform Adaptive KDE smoothing on a given dataset.

    Attributes:
        data (GeoDataFrame): The input data.
        params (dict): Dictionary of parameters.
        c (float): A constant for bandwidth calculation.
        k (int): Number of nearest neighbors to consider.
        test_data (GeoDataFrame): Test data for validation.
    """

    def __init__(self, data, params, c, k, shapefile=None, test_data=None, no_grid=True):
        super().__init__(data, params, shapefile, test_data, no_grid=no_grid)
        self.c = c
        self.k = k

    def model_knn_distances(self, x, y, k): 
        """
        Fits a NearestNeighbors model to the data.
        
        Parameters:
            x (ndarray): X coordinates.
            y (ndarray): Y coordinates.
            k (int): Number of nearest neighbors.

        Returns:
            knn: A KNN model.
        """
        if len(x) < k:
            # If the number of data points is less than k, consider all of them. TODO: Add revised numbers in the draft.
            k = len(x)
        knn = NearestNeighbors(n_neighbors = k)
        data = np.vstack((x, y)).T
        knn.fit(data)
        return knn

    def kde_wrapper(self, x, y, z, xnew, ynew, c, k):
        """
        Wrapper function for KDE smoothing.

        Parameters:
            x (ndarray): X coordinates.
            y (ndarray): Y coordinates.
            z (ndarray): Metric values.
            xnew (ndarray): New X coordinates.
            ynew (ndarray): New Y coordinates.
            c (float): A constant for bandwidth calculation.
            k (int): Number of nearest neighbors.

        Returns:
            z_smooth (ndarray): Smoothed metric values.
            w_smooth (ndarray): Smoothed weights.
        """
        knn = self.model_knn_distances(x, y, k)  # Fit a nearest neighbor model to sampled points
        if knn is None:
            return None, None

        new_coords = np.vstack([xnew, ynew]).T
        distances, indices = knn.kneighbors(new_coords, return_distance=True)

        # Precompute constants
        small_const = 1e-12
        z_smooth = np.zeros(len(xnew))

        # Vectorize distance and bandwidth calculation
        mean_distances = np.mean(distances, axis=1)
        bandwidths = small_const + c * (mean_distances ** 2)

        st = time.time()

        for i in range(len(xnew)):
            bw = bandwidths[i]
            kde = KernelDensity(bandwidth=bw, kernel='gaussian')
            x_nb, y_nb = x[indices[i]], y[indices[i]]
            coords = np.vstack((x_nb, y_nb)).T
            kde.fit(coords)
            w_new = np.exp(kde.score_samples(coords))  # Calculate neighbor weights
            z_new = np.sum(w_new * z[indices[i]]) / np.sum(w_new)  # Weighted average of the metric values for neighbors
            z_smooth[i] = z_new
        w_smooth = np.ones(len(z_smooth))  # Placeholder for weights, to be implemented if needed
        return z_smooth, w_smooth

    def smooth(self, h3_cell, xnew=None, ynew=None):
        """
        Performs Adaptive KDE smoothing for a specific H3 hexagon cell.

        Parameters:
            h3_cell (str): H3 hexagon cell identifier.

        Returns:
            z_smooth (ndarray): Smoothed metric values.
            w_smooth (ndarray): Smoothed weights.
        """
        data = self.data[self.data['hex_id'] == h3_cell]
        data = data.to_crs(self.epsg)

        x = data.geometry.x.values
        y = data.geometry.y.values
        z = data[self.metric].values

        isp_train = data['attr_provider_name'].values
        isp_test = self.test_data['attr_provider_name'].values

        if self.test_data is None:
            # Generate grid points within the hexagon. These will be used later for interpolation.
            grid_x, grid_y = generate_grid_points(h3_cell, self.epsg, self.grid_spacing)
            grid_z = np.full(len(grid_x), np.nan) # because we don't have ground truth for synthetic points
        else:
            # When we provide our own test data
            td_within = containment_filter(h3_cell, self.test_data)
            td_within = td_within.to_crs(self.epsg)
            grid_x = td_within.geometry.x.values
            grid_y = td_within.geometry.y.values
            grid_z = td_within[self.metric].values

        z_smooth, w_smooth = self.kde_wrapper(x, y, z, xnew=grid_x, ynew=grid_y, c=self.c, k=self.k)
        if z_smooth is None:
            return grid_x, grid_y, grid_z, None, None, None

        coeff = np.full(z_smooth.shape, np.nan) # For compatibility -- there are no coefficients here.

        return grid_x, grid_y, grid_z, z_smooth, w_smooth, coeff # values to be returned for the gridded hexagon

    def smooth_regionwide(self):
        """
        Performs Adaptive KDE smoothing over the entire region.

        Parameters:
            shapefile (str): Path to the shapefile.
        """
        data = self.data.copy()
        data = data.to_crs(self.epsg)

        if self.no_grid:
            xpt = self.test_data.geometry.x.values
            ypt = self.test_data.geometry.y.values
            zpt = self.test_data[self.metric].values
            grid_points = (xpt, ypt)
        else:
            grid_points = generate_grid_over_data(data, self.epsg, self.grid_spacing, self.shapefile)
            zpt = np.full(len(grid_points[0]), np.nan) # because we don't have ground truth for synthetic points

        x = data.geometry.x.values
        y = data.geometry.y.values
        z = data[self.metric].values

        z_smooth, w_smooth = self.kde_wrapper(x, y, z, xnew=grid_points[0], ynew=grid_points[1], c=self.c, k=self.k)

        return grid_points[0], grid_points[1], zpt, z_smooth, w_smooth

class LOESS_Smoother(Smoother):
    """
    A class to perform LOESS smoothing on a given dataset.

    Attributes:
        data (GeoDataFrame): The input data.
        params (dict): Dictionary of parameters.
        span (float): Fraction of data points to consider in each local regression.
        degree (int): Degree of the polynomial to fit.
        test_data (GeoDataFrame): Test data for validation.
    """

    def __init__(self, data, params, span, degree,shapefile=None, test_data=None, no_grid=True):
        super().__init__(data, params, shapefile=shapefile, test_data=test_data, no_grid=no_grid)
        self.span = span
        self.degree = degree

    def smooth(self, h3_cell, xnew=None, ynew=None):
        """
        Performs LOESS smoothing for a specific H3 hexagon cell.

        Parameters:
            h3_cell (str): H3 hexagon cell identifier.

        Returns:
            z_smooth (ndarray): Smoothed metric values.
            w_smooth (ndarray): Smoothed weights.
        """
        data = self.data[self.data['hex_id'] == h3_cell]
        data = data.to_crs(self.epsg)

        x = data.geometry.x.values
        y = data.geometry.y.values
        z = data[self.metric].values

        if self.test_data is None:
            # Generate grid points within the hexagon. These will be used later for interpolation.
            grid_points = generate_grid_points(h3_cell, self.epsg, self.grid_spacing)
            grid_x = np.array([point[0] for point in valid_grid_points])
            grid_y = np.array([point[1] for point in valid_grid_points])
            grid_z = np.full(len(grid_x), np.nan) # because we don't have ground truth for synthetic points
        else:
            # When we provide our own test data
            td_within = containment_filter(h3_cell, self.test_data)
            td_within = td_within.to_crs(self.epsg)
            grid_x = td_within.geometry.x.values
            grid_y = td_within.geometry.y.values
            grid_z = td_within[self.metric].values
        try:
            z_smooth, w_smooth = loess_2d(x, y, z, xnew=grid_x, ynew=grid_y, frac=self.span, degree=self.degree)
        except np.linalg.LinAlgError:
            # Case when SVD does not converge on least squares. We will discard these hexagons for now because interpolation will take care of them. 
            # TODO: A known fallback is simple linear regression; can be implemented later.
            return grid_x, grid_y, grid_z, None, None, None
        return grid_x, grid_y, grid_z, z_smooth, w_smooth, None # values to be returned for the gridded hexagon
    
    def smooth_regionwide(self):
        """
        Performs LOESS smoothing over the entire region.

        Parameters:
            shapefile (str): Path to the shapefile.
        """

        data = self.data.copy()
        data = data.to_crs(self.epsg)

        if self.no_grid:
            xpt = self.test_data.geometry.x.values
            ypt = self.test_data.geometry.y.values
            zpt = self.test_data[self.metric].values
            grid_points = (xpt, ypt)
        else:
            grid_points = generate_grid_over_data(data, self.epsg, self.grid_spacing, self.shapefile)
            zpt = np.full(len(grid_points[0]), np.nan) # because we don't have ground truth for synthetic points

        x = data.geometry.x.values
        y = data.geometry.y.values
        z = data[self.metric].values
        
        try:
            z_smooth, w_smooth = loess_2d(x, y, z, xnew=grid_points[0], ynew=grid_points[1], frac=self.span, degree=self.degree)
        except np.linalg.LinAlgError:
            # Case when SVD does not converge on least squares. We will discard these hexagons for now because interpolation will take care of them. 
            # TODO: A known fallback is simple linear regression; can be implemented later.
            return grid_points[0], grid_points[1], zpt, None, None

        return grid_points[0], grid_points[1], zpt, z_smooth, w_smooth

class IDW(Smoother):
    def __init__(self, data, params, p, shapefile=None, test_data=None, no_grid=True):
        super().__init__(data, params, shapefile=shapefile, test_data=test_data, no_grid=no_grid)
        self.p = p

    def predict(self, x0, y0, z0, x1, y1):
        """ 
        Predicts the values at the unsampled locations
        
        Returns:
            np.array: Predictions at the unsampled locations
        """
        predictions = []
        for i in range(len(x1)):
            dist = np.sqrt((x0 - x1[i])**2 + (y0 - y1[i])**2)
            w = 1 / (dist + 1e-12)**self.p
            w /= w.sum()
            predictions.append(np.dot(w, z0))
        return np.array(predictions), np.full(len(x1), np.nan)

    def smooth(self, h3_cell, xnew=None, ynew=None):
        """
        Performs IDW smoothing for a specific H3 hexagon cell.

        Parameters:
            h3_cell (str): H3 hexagon cell identifier.

        Returns:
            z_smooth (ndarray): Smoothed metric values.
            w_smooth (ndarray): Smoothed weights.
        """
        data = self.data[self.data['hex_id'] == h3_cell]
        data = data.to_crs(self.epsg)

        x = data.geometry.x.values
        y = data.geometry.y.values
        z = data[self.metric].values

        if self.test_data is None:
            # Generate grid points within the hexagon. These will be used later for interpolation.
            grid_points = generate_grid_points(h3_cell, self.epsg, self.grid_spacing)
            grid_x = np.array([point[0] for point in valid_grid_points])
            grid_y = np.array([point[1] for point in valid_grid_points])
            grid_z = np.full(len(grid_x), np.nan) # because we don't have ground truth for synthetic points
        else:
            # When we provide our own test data
            td_within = containment_filter(h3_cell, self.test_data)
            td_within = td_within.to_crs(self.epsg)
            grid_x = td_within.geometry.x.values
            grid_y = td_within.geometry.y.values
            grid_z = td_within[self.metric].values
        z_smooth, w_smooth = self.predict(x, y, z, grid_x, grid_y)
        coeff = np.full(z_smooth.shape, np.nan)

        return grid_x, grid_y, grid_z, z_smooth, w_smooth, coeff # values to be returned for the gridded hexagon

    def smooth_regionwide(self):
        """
        Performs IDW smoothing over the entire region.

        Parameters:
            shapefile (str): Path to the shapefile.
        """

        data = self.data.copy()
        data = data.to_crs(self.epsg)

        if self.no_grid:
            xpt = self.test_data.geometry.x.values
            ypt = self.test_data.geometry.y.values
            zpt = self.test_data[self.metric].values
            grid_points = (xpt, ypt)
        else:
            grid_points = generate_grid_over_data(data, self.epsg, self.grid_spacing, self.shapefile)
            zpt = np.full(len(grid_points[0]), np.nan) # because we don't have ground truth for synthetic points

        x = data.geometry.x.values
        y = data.geometry.y.values
        z = data[self.metric].values

        z_smooth, w_smooth = self.predict(x, y, z, x1=grid_points[0], y1=grid_points[1])
        return grid_points[0], grid_points[1], zpt, z_smooth, w_smooth
