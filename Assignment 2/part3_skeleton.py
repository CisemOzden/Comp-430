import numpy as np
import statistics
import pandas as pd
import math
import random

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


# You can define your own helper functions here. #

def calculate_average_error(actual_hist, noisy_hist):
    err = 0
    for i in range(len(actual_hist)):
        err += abs(actual_hist[i]-noisy_hist[i])
    return err/len(actual_hist)
### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    d = len(DOMAIN)
    p = math.exp(epsilon)/(math.exp(epsilon) + d - 1)
    q = (1 - p) / (d - 1)
    prob_list = [q]*d
    val_index = DOMAIN.index(val)
    prob_list[val_index] = p
    return np.random.choice(DOMAIN, 1, p = prob_list)[0]


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    freq_dict = {}
    counts_hist = [0]*len(DOMAIN)
    d = len(DOMAIN)
    p = math.exp(epsilon)/(math.exp(epsilon) + d - 1)
    q = (1 - p) / (d - 1)
    
    for val in perturbed_values:
        if val in freq_dict:
            freq_dict[val] +=1
        else:
            freq_dict[val] = 1
    for v in freq_dict.keys():
        count_estimate = (freq_dict[v] - (len(perturbed_values))*q)/(p - q) #estimator
        counts_hist[v-1] = count_estimate
    
    return counts_hist


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    actual_hist = [0]*len(DOMAIN)
    for i in range(len(DOMAIN)):
        count = dataset.count(i+1)
        actual_hist[i] = count
    perturbed_values = []
    for val in dataset:
        perturbed_values.append(perturb_grr(val, epsilon))
    estimated_hist = estimate_grr(perturbed_values, epsilon)
    
    return calculate_average_error(actual_hist, estimated_hist)


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    bit_vector = [0]*len(DOMAIN)
    bit_vector[val -1] = 1
    return bit_vector


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    perturbed_vector = []
    p = math.exp(epsilon/2)/(math.exp(epsilon/2) + 1)
    for i in range(len(encoded_val)):
        if random.random() < p:
            perturbed_vector.append(encoded_val[i])
        else:
            perturbed_vector.append(abs(encoded_val[i] - 1))
    return perturbed_vector


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    p = math.exp(epsilon/2)/(math.exp(epsilon/2) + 1)
    q = 1 - p
    sum_vector = np.sum(perturbed_values, axis=0).tolist()
    estimate_vector = []
    for i in range(len(sum_vector)):
        count_estimate = (sum_vector[i] - len(perturbed_values)*q) / (p - q)
        estimate_vector.append(count_estimate)
    return estimate_vector


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    actual_hist = [0]*len(DOMAIN)
    for i in range(len(DOMAIN)):
        count = dataset.count(i+1)
        actual_hist[i] = count
    perturbed_values = []
    for val in dataset:
        perturb_vector = perturb_rappor(encode_rappor(val), epsilon)
        perturbed_values.append(perturb_vector)
    estimated_hist = estimate_rappor(perturbed_values, epsilon)
    
    return calculate_average_error(actual_hist, estimated_hist)


# OUE

# TODO: Implement this function!
def encode_oue(val):
    bit_vector = [0]*len(DOMAIN)
    bit_vector[val -1] = 1
    return bit_vector


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    perturbed_vector = []
    p1 = 0.5 #1->1
    p2 = math.exp(epsilon)/(math.exp(epsilon) + 1) #0->0
    for i in range(len(encoded_val)):
        if encoded_val[i] == 1:
            if random.random() < p1:    
                perturbed_vector.append(1)
            else:
                perturbed_vector.append(0)
        else:
            if random.random() < p2:
                perturbed_vector.append(0)   
            else:
                perturbed_vector.append(1)
    return perturbed_vector


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    sum_vector = np.sum(perturbed_values, axis=0).tolist()
    estimate_vector = []
    for i in range(len(sum_vector)):
        count_estimate = 2*((math.exp(epsilon) + 1)*sum_vector[i] - len(perturbed_values)) / (math.exp(epsilon) - 1)
        estimate_vector.append(count_estimate)
    return estimate_vector


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    actual_hist = [0]*len(DOMAIN)
    for i in range(len(DOMAIN)):
        count = dataset.count(i+1)
        actual_hist[i] = count
    perturbed_values = []
    for val in dataset:
        perturb_vector = perturb_oue(encode_oue(val), epsilon)
        perturbed_values.append(perturb_vector)
    estimated_hist = estimate_oue(perturbed_values, epsilon)
    
    return calculate_average_error(actual_hist, estimated_hist)


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")

    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()

