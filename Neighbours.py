# k-Nearest assignment

"""
The data is in the following format
(date);windspeed;temp_avg;temp_min;temp_max;suntime;raintime;rain_amt

"""
from typing import List, Any, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_weather_data(dset: np.ndarray[int, Any],
                           *other_sets: np.ndarray[[int, Any]]) -> Union[
    np.ndarray[float, Any], Tuple[np.ndarray[float, Any], Any]]:
    """
    The weather data needs to be normalized.

    The lows and highs of all datasets are combined and used, so that all the data points are equally normalized.
    the following tactic is used for normalizing:
    (X - Xmin) / (Xmax - Xmin)
    :param dset: the un-normalized dataset. (np.ndarray[float, Any])
    :param other_sets: optional other datasets. (List[np.array[int, Any]])
    :return: the normalized dataset(s). (Union[pd.DataFrame, Tuple[pd.Dataframe]])
    """

    ndset: np.ndarray[float, Any]
    other_ndsets: List[np.ndarray[float, Any]] = list()

    # getting all the columns min and maxes
    dset = np.delete(dset, [0], axis=1)

    mins: np.array = np.array([dset.min(0)])
    maxes: np.array = np.array([dset.max(0)])

    for other_set in other_sets:
        other_set = np.delete(other_set, [0], axis=1)
        min = np.array([other_set.min(0)])
        max = np.array([other_set.max(0)])
        mins = np.append(mins, min, axis=0)
        maxes = np.append(maxes, max, axis=0)

    # apply normalization techniques
    if other_sets is None:
        ndset = np.subtract(dset, mins.min(0)) / np.subtract(maxes.max(0), mins.min(0))
        return ndset
    else:
        ndset = np.subtract(dset, mins.min(0)) / np.subtract(maxes.max(0), mins.min(0))
        for other_dset in other_sets:
            other_dset = np.delete(other_dset, [0], axis=1)
            other_ndset = np.subtract(other_dset, mins.min(0)) / np.subtract(maxes.max(0), mins.min(0))
            other_ndsets.append(other_ndset)
        return ndset, *other_ndsets


def add_dates(dateless: list, dates) -> Tuple:
    """
    The already normalised functions now lack a date, so it should be added again.
    Keep in mind that the order of the datasets in both params should be in the same order
    :param dateless: list of normalized (dateless) datasets.
    :param dates: list of original datasets containing the dates.
    :return: the normalized datasets with dates.
    """
    result: List = list()
    for i, df in enumerate(dateless):
        normalized = pd.DataFrame(dateless[i], columns=['FG', 'TG', 'TN', 'TX', 'SQ', 'DR', 'RH'])
        normalized.insert(0, 'Date',
                          pd.DataFrame(dates[i], columns=['Date', 'FG', 'TG', 'TN', 'TX', 'SQ', 'DR', 'RH'])[
                              'Date'])
        print(f"Dataset:\n"
              f"{normalized}")
        result.append(normalized.to_numpy())
    return result[0], *result[1:]


def calculate_distance(a, b) -> float:
    """
    calculate the euclidean distance
    :param a: first list containing data
    :param b: second list containing data
    :return: the euclidian distance as a float.
    """
    return np.sqrt(np.sum((a - b) ** 2))


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
        # return the closest neighbour :)
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


def KNN(training_dataset: np.array, test_dataset: np.array, hvar_k) -> List[str]:
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
        training_dataset = sorted(training_dataset, key=lambda x: calculate_distance(weatherdata[1:], x[1:]))
        solved.append([get_season(training_dataset, hvar_k), weatherdata])
    return solved


if __name__ == '__main__':

    ################### PART ONE ####################

    dataset1: np.ndarray[int, Any] = np.genfromtxt('datasets/dataset1.csv', delimiter=';',
                                                   usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    data1: np.ndarray[int, Any] = np.genfromtxt('datasets/days.csv', delimiter=';', usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                                converters={})
    validation1: np.ndarray[int, Any] = np.genfromtxt('datasets/validation1.csv', delimiter=';',
                                                      usecols=[0, 1, 2, 3, 4, 5, 6, 7])

    # Variable for the different accuracy rates.
    accuracy_points = []

    for k in range(1, 100):
        # Create a variable to keep track of successful season assumptions
        correct_assumpt = 0
        result = KNN(dataset1, data1, k)
        # get the assumed seasons from KNN.
        seasons = list(row[:][0] for row in result)
        for row_iter in range(len(seasons)):
            if translate(validation1[row_iter][0]) == seasons[row_iter]:
                correct_assumpt += 1
        accuracy_points.append(correct_assumpt)

    # Plot Accuracy
    plt.xlabel('value of k -->')
    plt.ylabel('accuracy in % -->')
    plt.plot(accuracy_points)
    plt.show()

    # in this plot, we see that without the normalisation,
    # an accuracy of 65 percent is reached.

    ################### PART TWO ####################

    # now for checking the days.

    # Create training_data with dates.
    dataset2: np.ndarray[int, Any] = np.genfromtxt('datasets/dataset1.csv', delimiter=';',
                                                   usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    data2: np.ndarray[int, Any] = np.genfromtxt('datasets/validation1.csv', delimiter=';',
                                                usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                                converters={})
    checkdates: np.ndarray[int, Any] = np.genfromtxt('datasets/days.csv', delimiter=';',
                                                     usecols=[0, 1, 2, 3, 4, 5, 6, 7])

    # normalize everything
    dataset2, data2, checkdates = add_dates([*normalize_weather_data(dataset2, data2, checkdates)],
                                            [dataset2, data2, checkdates])

    # Variable for the different accuracy rates.
    accuracy_points = []

    for k in range(1, 350):
        # Create a variable to keep track of successful season assumptions
        correct_assumpt = 0
        result = KNN(dataset2, data2, k)
        # get the assumed seasons from KNN.
        seasons = list(row[:][0] for row in result)
        for row_iter in range(len(seasons)):
            if translate(checkdates[row_iter][0]) == seasons[row_iter]:
                correct_assumpt += 1
        accuracy_points.append(correct_assumpt)

    # Plot Accuracy
    plt.xlabel('value of k -->')
    plt.ylabel('accuracy in % -->')
    plt.plot(accuracy_points)
    plt.show()

    # in this plot, the best accuracy is 62,5 percent.
    # Note that this time the normalisation is applied to the datasets,
    # but the accuracy per k amount of neighbours does not change significantly.
