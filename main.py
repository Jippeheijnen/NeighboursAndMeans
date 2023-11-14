# k-Nearest assignment

"""
The data is in the following format
(date);windspeed;temp_avg;temp_min;temp_max;suntime;raintime;rain_amt

"""
import math
from typing import List, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

"""
    
    :param dset: the un-normalized dataset
    :return: the same dataset but normalized.
    """


# todo: implement this
def normalize_weather_data(dset: np.ndarray[int, Any],
                           *other_sets: np.ndarray[[int, Any]]) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    The weather data needs to be normalized.
    the following tactic is used for normalizing:
    (X - Xmin) / (Xmax - Xmin)
    :param dset: the un-normalized dataset. (np.ndarray[int, Any])
    :param other_sets: optional other datasets. (List[np.array[int, Any]])
    :return: the normalized dataset(s). (Union[pd.DataFrame, Tuple[pd.Dataframe]])
    """

    ndset = pd.DataFrame(dset.copy())
    other_ndsets: List[pd.DataFrame] = []

    # apply normalization techniques
    if other_sets is None:
        for column in ndset.columns:
            ndset[column] = (ndset[column] -
                             ndset[column].min()) / ndset[column].max()
        return ndset
    else:
        for column in ndset.columns:
            ndset[column] = (ndset[column] -
                             ndset[column].min()) / ndset[column].max()
        for other_dset in other_sets:
            other_ndset: pd.DataFrame = pd.DataFrame(other_dset.copy())
            for column in other_ndset.columns:
                other_ndset[column] = (other_ndset[column] -
                                       other_ndset[column].min()) / other_ndset[column].max()
            other_ndsets.append(other_ndset)
        return ndset, *other_ndsets


def calculate_distance(a, b) -> float:
    """
    calculate the euclidean distance
    :param a: first list containing data
    :param b: second list containing data
    :return: the euclidian distance as a float.
    """
    return math.dist(a, b)


def translate(date):
    """
    Translate date to a season.
    :param date: The date as integer, format %Y%m%d.
    :return: Returns a string with the season.
    """
    date_as_datetime = pd.to_datetime(str(int(date)), format='%Y%m%d')
    if date_as_datetime.month < 3:
        return "winter"
    elif 3 <= date_as_datetime.month < 6:
        return "lente"
    elif 6 <= date_as_datetime.month < 9:
        return "zomer"
    elif 9 <= date_as_datetime.month < 12:
        return "herfst"
    else:
        return "winter"


def get_season(dataset, hvar_k):
    """
    This function returns the season for a datapoint as a string.
    :param dataset: the dataset to iterate over. Note: the datapoints must contain a date at location [0].
    :param hvar_k: Hypervariable K.
    :return: Season as a string.
    """

    # the season counters; w for winter l for lente etc.
    winter_cnt, lente_cnt, zomer_cnt, herfst_cnt = 0, 0, 0, 0
    for entry_index in range(0, hvar_k):
        # [0] to get the date part
        season_string = translate(dataset[entry_index][0])
        if season_string == 'winter':
            winter_cnt += 1
        elif season_string == 'lente':
            lente_cnt += 1
        elif season_string == 'zomer':
            zomer_cnt += 1
        elif season_string == 'herfst':
            herfst_cnt += 1

    # check whether there is a tie.
    if (winter_cnt == lente_cnt or winter_cnt == zomer_cnt or winter_cnt == herfst_cnt) or (
            lente_cnt == zomer_cnt or lente_cnt == herfst_cnt) or (zomer_cnt == herfst_cnt):
        # return the closed neighbour :)
        return translate(dataset[0][0])

    # check which season occurs most in neighbouring datapoints, and return that season.
    season_count = max([winter_cnt, lente_cnt, zomer_cnt, herfst_cnt])
    if season_count == winter_cnt:
        return "winter"
    elif season_count == lente_cnt:
        return "lente"
    elif season_count == zomer_cnt:
        return "zomer"
    elif season_count == herfst_cnt:
        return "herfst"


def KNN(training_dataset, test_dataset, hvar_k) -> List[str]:
    """
    The general k-nearest-neighbours algorithm.
    :param training_dataset: The dataset in which known weather data is collected.
    :param test_dataset: The 'unknown' weather data.
    :param hvar_k: Hypervariable k
    :return: a list of seasons from the test_dataset.
    """
    solved: List[Union[str, Any]] = []
    for weatherdata in test_dataset:
        # Now we will sort the dataset by least euclidian distance to the current weatherdata.
        # This way the least distances will be at the front and now we only
        # have to iterate k times over the dataset.

        # x[1:] because the date is stored in 'dataset' as well.
        training_dataset = sorted(training_dataset, key=lambda x: calculate_distance(weatherdata, x[1:]))
        solved.append([get_season(training_dataset, hvar_k), weatherdata])
    return solved


if __name__ == '__main__':

    ################### PART ONE ####################

    dataset1: np.ndarray[int, Any] = np.genfromtxt('datasets/dataset1.csv', delimiter=';',
                                                   usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                                   converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                               7: lambda s: 0 if s == b"-1" else float(s)})
    data1: np.ndarray[int, Any] = np.genfromtxt('datasets/days.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                                                converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                            7: lambda s: 0 if s == b"-1" else float(s)})

    # choose value for k.
    k = 3

    # normalize the datasets.
    # dataset1: pd.DataFrame = normalize_weather_data(dataset1)
    # data1: pd.DataFrame = normalize_weather_data(data1)

    dataset1, data1 = normalize_weather_data(dataset1, data1, )

    print(f"Dataset1:\n"
          f"{dataset1}")
    print(f"Data1:\n"
          f"{data1}")

    # print the assigned seasons
    print(f"The 9 assumed seasons for days.csv\n"
          f"{list(list[:][0] for list in KNN(dataset1, data1, k))}"
          f"\n")

    ################### PART TWO ####################

    # now for checking the validation list.

    # Create training_data with dates.
    dataset2 = np.genfromtxt('datasets/dataset1.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                             converters={5: lambda s: 0 if s == b"-1" else float(s),
                                         7: lambda s: 0 if s == b"-1" else float(s)})
    # Create test_data without dates.
    data2 = np.genfromtxt('datasets/validation1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                          converters={5: lambda s: 0 if s == b"-1" else float(s),
                                      7: lambda s: 0 if s == b"-1" else float(s)})

    # list with dates to validate the assumed seasons
    checkdates = np.genfromtxt('datasets/validation1.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                               converters={5: lambda s: 0 if s == b"-1" else float(s),
                                           7: lambda s: 0 if s == b"-1" else float(s)})

    # Variable for the different accuracy rates.
    accuracy_points = []

    # Running this takes a minute or two, so I included the png file.
    for k in range(1, 100):
        # Create a variable to keep track of successful season assumptions
        correct_assumpt = 0

        # get the assumed seasons from KNN.
        seasons = list(row[:][0] for row in KNN(dataset2, data2, k))
        for row_iter in range(len(seasons)):
            if translate(checkdates[row_iter][0]) == seasons[row_iter]:
                correct_assumpt += 1
        accuracy_points.append(correct_assumpt)

    # Plot Accuracy
    plt.xlabel('value of k -->')
    plt.ylabel('accuracy in % -->')
    plt.plot(accuracy_points)
    plt.show()
