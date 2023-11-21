#
# Created by Jippe Heijnen on 20-11-2023.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import copy
import random

import numpy as np
from typing import Any, List, Union

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from Neighbours import normalize_weather_data, add_dates, calculate_distance


def validate_clusters(clusters: List[List[np.ndarray]], centroids: np.ndarray) -> float:
    """
    This function calculates an error margin by dividing the total cum distance from
    all cluster points to the cluster centroid, by the total amount of points combined.
    :param clusters: The complete clusterset.
    :param centroids: the centroids of each cluster.
    :return: a floating point, between 0 and 1. It represents the cum distance divided by the total nodes.
    """
    distance_from_centroids: float = 0
    sum_of_points: int = sum([len(x) for x in clusters])

    for clus_id, centroid in enumerate(centroids):
        for point in clusters[clus_id]:
            distance_from_centroids += calculate_distance(point, centroid)

    return distance_from_centroids / sum_of_points


def accuracy(dset: np.ndarray, k: int, max_iter: int = 10) -> float:
    """
    This function returns the lowest error margin by running the kMeans algorithm multiple times.
    :param dset: the dataset with points.
    :param k: amount of clusters.
    :param max_iter: Max times of margin calculations.
    :return: Lowest error margin for this k.
    """
    margin = np.inf

    for _ in range(max_iter):
        centroids = pick_centroids(dset, k)
        clusters, centroids = means(dset, centroids)
        curr_margin = validate_clusters(clusters, centroids)
        if curr_margin < margin:
            margin = curr_margin
    return margin


def pick_centroids(dset: np.ndarray, k: int) -> np.ndarray:
    """
    This function returns an array of randomized centroids for an amount of k.
    It also checks if each centroid is unique to its predecessors.
    :param dset: The dataset containing points.
    :param k: Amount of centroids that will be picked.
    :return: The chosen centroids, consisting of random points from dset.
    """
    centroids: np.array = dset[np.random.choice(len(dset), 1)]

    # making sure the centroids are unique and not duplicate
    for _ in range(k - 1):
        attempt = dset[np.random.choice(len(dset), 1)]
        while attempt[0] in centroids:
            attempt = dset[np.random.choice(len(dset), 1)]
        centroids = np.append(centroids, attempt, axis=0)
        centroids.sort()
    return centroids


def assign_labels(dset: np.array, centroids: np.array) -> np.array:
    """
    This function will assign labels to each datapoint, according to its distance to each centroid.
    :param dset: The dataset containing points.
    :param centroids: array of centroids.
    :return: The whole dataset with labels at the index of its corresponding datapoint.
    """

    # creating a matrix of distances per point per centroid
    distances = np.full(shape=(len(dset), len(centroids)), fill_value=np.inf)

    for i, c in enumerate(centroids):
        distances[:, i] = [calculate_distance(x, c) for j, x in enumerate(dset)]

    labels = np.argmin(distances, axis=1)

    return labels


def means(dset: np.array, centroids: np.array) -> Union[List[List[np.ndarray]], np.ndarray]:
    """
    This is the kMeans algorithm.
    1: It iterates through the dataset, and for each datapoint it calculates the distance to all centroids.
    2: Each datapoint will be assigned to the cluster with the closest centroid to the datapoint.
    3: Then the centroids are refreshed for all clusters and it starts again.
    4: for each loop, an equality check occurs between the previous centroids and the new centroids. If they are equal,
    it means no datapoint switched clusters, and the looping can be stopped.
    :param dset: The dataset containing the datapoints.
    :param centroids: The array of centroids.
    :return: The completed clusterset and the according centroids.
    """

    k = len(centroids)

    clusters = [[] for x in range(k)]

    # setting random initial centroids
    prev_centroids = copy.deepcopy(centroids)

    # filing the clusterset with epmty clusters

    for point in dset:
        closest_centroid = (np.inf, 0)

        # calculating closest centroids
        for cent_id, centroid in enumerate(centroids):
            distance: float = 0
            distance += calculate_distance(point, centroid)
            if distance < closest_centroid[0]:
                closest_centroid = (distance, cent_id)

        # Adding point to cluster
        clusters[closest_centroid[1]].append(point)

    for clus_id, cluster in enumerate(clusters):
        if not len(cluster) == 0:
            mean = np.array(cluster).mean(axis=0)
            centroids[clus_id] = mean

    if np.array_equal(prev_centroids, centroids):
        return clusters, centroids
    else:
        return means(dset, centroids)


if __name__ == '__main__':
    randomness_seed = 101010
    np.random.seed = randomness_seed

    # Create training_data with dates.
    dataset: np.ndarray[int, Any] = np.genfromtxt('datasets/dataset1.csv', delimiter=';',
                                                  usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    validation: np.ndarray[int, Any] = np.genfromtxt('datasets/validation1.csv', delimiter=';',
                                                     usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                                     converters={})

    dataset_dateless, validation_dateless = normalize_weather_data(dataset, validation)
    dataset_dates, validation_dates = add_dates([dataset_dateless, validation_dateless], [dataset, validation])

    # and now for the plot

    k_max = 15

    margins = [accuracy(dataset_dateless, k + 1, 10) for k in range(k_max)]

    plt.plot(
        [k + 1 for k in range(k_max)],
        margins,
        'o-',
        linewidth=2,
        color='red',
        marker='o',
        markeredgecolor='#1e78b4',
        markerfacecolor='#1e78b4',
        markersize=5)

    plt.title("K means error margins")
    plt.xlabel("K")
    plt.ylabel("Error margin")
    plt.show()
