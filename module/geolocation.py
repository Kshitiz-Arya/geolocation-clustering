from functools import lru_cache
from os import getenv

import hdbscan
import numpy as np
import openrouteservice
import pandas as pd
from dotenv import load_dotenv
from h3 import h3
from openrouteservice.distance_matrix import distance_matrix
from shapely.geometry import MultiPoint


class analyse:

    def __init__(self, data, outlier_threshold=0.8, h3_level=12, **kwargs):

        self._outlier_threshold = outlier_threshold
        self._h3_level = h3_level
        self._data = self.__time_spent(data)
        self._epsilon_threshold = kwargs.get("epsilon_hreshold", 0.005)

        self._min_cluster_size = kwargs.get("min_cluster_size", 10)
        self._min_samples = kwargs.get("min_samples", None)
        self._metric = kwargs.get("metric", "haversine")
        self._alpha = kwargs.get("alpha", 1.0)
        self._algorithm = kwargs.get("algorithm", "best")
        self._leaf_size = kwargs.get("leaf_size", 40)
        self._approx_min_span_tree = kwargs.get("approx_min_span_tree", True)
        self._gen_min_span_tree = kwargs.get("gen_min_span_tree", False)
        self._core_dist_n_jobs = kwargs.get("core_dist_n_jobs", 4)
        self._cluster_selection_method = kwargs.get("cluster_selection_method", "leaf")
        self._allow_single_cluster = kwargs.get("allow_single_cluster", False)

        load_dotenv()

    def __time_spent(self, data):
        # Change this to numpy equivaleant
        timespend = (data["timestamp"].diff(1) / 2).fillna(0)
        data["timespend"] = timespend + timespend.shift(-1).fillna(0)
        return data.copy()

    def __refit(self, unique, original):
        """This method merge two dataframes in one with [longiude, latitude] as the common column. One dataframe has unique
        [latitude, longitude], whereas the second one contains duplicate values

        Args:
            unique (panda dataframe): This dataframe must have [latitude, longitude, label, outlier, hex] as columns
            original (pandas dataframe): The original dataframe that was passed to the class

        Returns:
            pandas dataframe: The dataframe after the merging of unique and original dataframe
        """

        unique[['latitude', 'longitude', 'label']] = unique[['latitude', 'longitude', 'label']].astype(float)
        original = pd.merge(original, unique, on=['latitude', 'longitude'])
        return original[original.label != -1].copy()

    def __cluster(self, data):
        """This method is responsible for the clustering of data

        Args:
            data (pandas dataframe): This is the original dataframe entered by the user

        Returns:
            pandas dataframe: This is the dataframe returned by self.__refit method
        """        
        
        # drop_duplicates: 13 ms ± 198 µs per loop ; np.unique : 93.3 ms ± 5.66 ms per loop
        geolocation = data[["latitude", "longitude"]].drop_duplicates().to_numpy()  # This make processing much easier and efficient
        geolocation_radians = np.radians(geolocation)   # LatLong needs to be converted in radians form to work with haversine matrix in HDBSCAN
        
        earth_radius_km = 6371
        epsilon = (
            self._epsilon_threshold / earth_radius_km
        )  # calculate epsilon threshold (default 5 m)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            metric=self._metric,
            alpha=self._alpha,
            algorithm=self._algorithm,
            leaf_size=self._leaf_size,
            approx_min_span_tree=self._approx_min_span_tree,
            gen_min_span_tree=self._gen_min_span_tree,
            core_dist_n_jobs=self._core_dist_n_jobs,
            cluster_selection_method=self._cluster_selection_method,
            allow_single_cluster=self._allow_single_cluster,
            cluster_selection_epsilon=epsilon,
        )
        clusterer.fit(geolocation_radians)

        clusterer.labels_[clusterer.outlier_scores_ > self._outlier_threshold] = -1 # Converting all points with outlier score more then the threshold value to noise

        hexes = self.__get_h3(geolocation, self._h3_level)  # Getting hex of unique latlong makes it much more efficient to process

        labels_outliers_stack = np.column_stack(
            (geolocation, clusterer.labels_, clusterer.outlier_scores_, hexes)
        )   # Creating the column stack of all the values, which will be later converted to pandas dataframe

        labels_outliers_hex = pd.DataFrame(data = labels_outliers_stack, columns=['latitude', 'longitude', 'label', 'outlier', 'hex'])
        return self.__refit(labels_outliers_hex, data)

    def __get_h3(self, geolocation, h3_level):
        """This method returns the hex of location points

        Args:
            geolocation (numpy array): This array should have location points in latlong format
            h3_level (int): This is the level of resolution to be used for getting hex

        Returns:
            list: List of hexes for each location points
        """        
        return [h3.geo_to_h3(row[0], row[1], h3_level) for row in geolocation]

    def __get_polygon(self, hexes):
        """This method merge multiple hexes into one polygon

        Args:
            hexes (h3 hex): list of hexes to merge into one polygon

        Returns:
            list: list of points representing a polygon
        """        
        # This method of merging multiple hexes have a downside, where if hexes aren't immediate neighbor then they will form multiple polygons
        return h3.h3_set_to_multi_polygon(hexes, geo_json=False) 

    def get_cluster(self, user=None, mode="mass"):
        data = (
            self.__cluster(self._data)
            if mode == "mass"
            else self.__cluster(self._data[self._data["user_id"] == user])
        )

        hexagons = data["hex"]
        clusters = data["label"].drop_duplicates()
        return {
            clust: {
                "polygon": self.__get_polygon(
                    hexagons[data["label"] == clust].drop_duplicates()
                ),
                "timespend": data.loc[data["label"] == clust, "timespend"].sum(),
            }
            for clust in clusters
        }


def get_path(polygons, destination):
    centroid = [
        list(MultiPoint(latlongs).centroid.coords)[0][::-1] for latlongs in polygons
    ]
    destination = [loc[::-1] for loc in destination]
    coordinate = centroid + destination
    request = {
        "locations": coordinate,
        "sources": [index for index in range(len(coordinate[: len(centroid)]))],
        "destinations": [
            index
            for index in range(
                len(centroid), len(centroid) + len(coordinate[len(centroid) :])
            )
        ],
        "metrics": ["distance"],
        "units": "km",
    }
    travel_modes = ["driving-car", "driving-hgv", "foot-walking", "cycling-regular"]
    routes = [
        distance_matrix(
            client=openrouteservice.Client(key=getenv("key")),
            **request,
            profile=travel_mode
        )["distances"]
        for travel_mode in travel_modes
    ]
    return list(zip(*routes))
