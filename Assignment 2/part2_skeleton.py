import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random

""" 
    Helper functions
    (You can define your helper functions here.)
"""


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    df = pd.read_csv(filename, sep=',', header = 0)
    return df


### HELPERS END ###


''' Functions to implement '''

# TODO: Implement this function!
def get_histogram(dataset, chosen_anime_id="199"):
    occ_dic = {-1:0, 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    res_list = []
    for val in dataset[chosen_anime_id]:
        if pd.isna(val):
            continue
        occ_dic[val] += 1
    for i in range(-1,11):
        res_list.append(occ_dic[i])
    return res_list

# TODO: Implement this function!
def get_dp_histogram(counts, epsilon: float):
    dp_list = []
    s = 1 #sensitivity
    b = s/epsilon #scale
    samples = np.random.laplace(0, b, size=len(counts)).tolist()
    noisy_hist = [sum(x) for x in zip(counts, samples)]
    return noisy_hist


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    err = 0
    for i in range(len(actual_hist)):
        err += abs(actual_hist[i]-noisy_hist[i])
    return err/len(actual_hist)

# TODO: Implement this function!
def calculate_mean_squared_error(actual_hist, noisy_hist):
    err = 0
    for i in range(len(actual_hist)):
        err += (actual_hist[i]-noisy_hist[i])**2
    return err/len(actual_hist)


# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    mean_sqr_errors = []
    avg_errors = []
    for j in range(len(eps_values)):
        total_avg_err = 0
        total_mean_sqr_err = 0
        for i in range(40):
            dp_hist = get_dp_histogram(counts, eps_values[j])
            avg_err = calculate_average_error(counts, dp_hist)
            mean_sqr_err = calculate_mean_squared_error(counts, dp_hist)
            total_avg_err += avg_err
            total_mean_sqr_err += mean_sqr_err
        avg_errors.append(total_avg_err/40)
        mean_sqr_errors.append(total_mean_sqr_err/40)
    return avg_errors, mean_sqr_errors


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    s = 1 #sensitivity
    returning_probs = {} #keeps the probability of returning corresponding key
    weights = []
    for col in dataset.columns[1:]:
        counts = get_histogram(dataset, col)
        q = counts[-1:][0] #gives the last element, number of 10 ratings
        weight = math.exp((epsilon*q)/(2*s))
        weights.append(weight)
    prob_list = [x / sum(weights) for x in weights]
    anime_ids = list(dataset.columns[1:])
    return np.random.choice(anime_ids, 1, p=prob_list)[0]


# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list): #just for your information, exponential experiment takes about 10 min
    max_10rate = -1
    max_col = ''
    for col in dataset.columns[1:]:
        counts = get_histogram(dataset, col)
        q = counts[-1:][0]
        if q > max_10rate:
            max_10rate = q
            max_col = col
 
    accuracy_list = []
    for j in range(len(eps_values)):
        count = 0
        for i in range(1000):
            val = most_10rated_exponential(dataset, eps_values[j])
            if val==max_col:
                count +=1
        accuracy_list.append(count/10) #append the accuracies
    return accuracy_list


# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "anime-dp.csv"
    dataset = read_dataset(filename)

    counts = get_histogram(dataset)

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg, error_mse = epsilon_experiment(counts, eps_values)
    print("**** AVERAGE ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])
    print("**** MEAN SQUARED ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_mse[i])


    print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])
        
#code to plot histogram
"""       
    ratings = range(-1,11)
    values = get_histogram(read_dataset("anime-dp.csv"), chosen_anime_id="199")

    fig = plt.figure(figsize = (6, 6))

    # creating the bar plot
    plt.bar(ratings, values, color ='blue',width = 0.9)
    plt.xticks(list(range(-1,11))) 
    plt.ylabel("Counts")
    plt.title("Rating Counts for Anime id=199")
    plt.show()
"""


if __name__ == "__main__":
    main()

