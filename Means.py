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
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from Neighbours import normalize_weather_data, add_dates, calculate_distance


def validate_clusters(clusters, centroids):
    """

    :param clusters:
    :param centroids:
    :return:
    """
    distance_from_centroids: float = 0
    sum_of_points: int = sum([len(x) for x in clusters])

    for clus_id, centroid in enumerate(centroids):
        for point in clusters[clus_id]:
            distance_from_centroids += calculate_distance(point, centroid)

    return distance_from_centroids/sum_of_points


def accuracy(dset, k, max_iter=100):
    margin = np.inf

    for _ in range(max_iter):
        centroids = pick_centroids(dset, k)
        clusters, centroids = means(dset, centroids)
        curr_margin = validate_clusters(clusters, centroids)
        if curr_margin < margin:
            margin = curr_margin
    return margin


def pick_centroids(dset: np.array, k: int):
    """

    :param dset:
    :param k:
    :return:
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


def assign_labels(dset: np.array, centroids: np.array, k: int) -> np.array:
    """

    :param dset:
    :param centroids:
    :param k:
    :return:
    """

    # creating a matrix of distances per point per centroid
    distances = np.full(shape=(len(dset), len(centroids)), fill_value=np.inf)

    for i, c in enumerate(centroids):
        distances[:, i] = [calculate_distance(x, c) for j, x in enumerate(dset)]

    if False and distances.shape > (len(dset), 2):  # start checking whether distances are equal
        for distance in distances:
            for c in centroids:
                pass
    else:
        labels = np.argmin(distances, axis=1)

    return labels


def means(dset: np.array, centroids: np.array) -> Any:
    """

    :param dset:
    :param centroids:
    :return:
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

    margins = [accuracy(dataset_dateless, k+1, 10) for k in range(k_max)]

    plt.plot(
        [k+1 for k in range(k_max)],
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