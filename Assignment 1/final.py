#!/usr/bin/env python
# coding: utf-8

# In[1]:


##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
from enum import unique
import glob
import os
import sys
from copy import deepcopy
import numpy as np
import datetime

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

import yaml
import pandas as pd
import itertools
from functools import reduce

find_generalized = lambda t_: reduce(lambda x,y:[i for i in x if i in y] ,t_)[-1]

def put_level(dic, level=0, dic2=None):
    if dic2==None:
        dic2=dict()
    for key in dic.keys():
        if dic[key] == None:
            dic2[key] = level
        else:
            dic2[key] = level
            put_level(dic[key], level+1, dic2)
    return dic2
            
def put_all_levels(dic):
    result = {}
    for key in dic.keys():
        result[key] = put_level(dic[key], 0)
    return result

def leaf_calculator(dic, count, dic2=None):
    if dic2==None:
        dic2=dict()
    for k in dic:
        if dic[k]==None:
            dic2[k] = 0
            count +=1
        else:
            count2 = 0
            c, d = leaf_calculator(dic[k],0,dic2)
            count2 += c
            count += c
            dic2[k] = count2
    return count, dic2

def leaf_calculator_all(dic):
    result = {}
    for key in dic.keys():
        c, result[key] = leaf_calculator(dic[key], 0)
    return result

def create_parents(dic, liste=None, parent_dict=None):
    if liste==None:
        liste = []
    if parent_dict==None:
        parent_dict = {}
    for k in dic:
        if dic[k]==None:
            temp = liste.copy()           
            liste.append(k)
            parent_dict[k] = liste.copy()
            liste = temp.copy()            
        else:
            temp = liste.copy()
            liste.append(k)
            create_parents(dic[k], liste, parent_dict)
            liste = temp.copy()
    return parent_dict
    
def parent_creator_all(dic):
    result = {}
    for key in dic.keys():
        result[key] = create_parents(dic[key])
    return result


def lm_cost(node, raw_dataset, parents_dict, leaf_dict):   
    total_cost = 0
    qi_count = len(raw_dataset[0])-1
    
    k_dict = {}
    for t in node: #iterate through node to match the corresponding levels to the fields
        k_dict[t[0]] = t[1]
        
    for i in range(len(raw_dataset)):   
        for field in raw_dataset[0].keys():    
            if field != 'income':
                k = k_dict[field]
                parent_size = len(parents_dict[field][raw_dataset[i][field]]) 
                general = parents_dict[field][raw_dataset[i][field]][parent_size-1-k]
                num1 = leaf_dict[field][general]
                lm_cost = 0
                if num1==0:
                    lm_cost = 0
                else:
                    num2 = leaf_dict[field]['Any']
                    lm_cost = (num1-1)/(num2-1)
                total_cost += lm_cost/qi_count               
    return total_cost

def binary_search(lattice_dic, low, high, raw_dataset, parents_dict, k, leaf_dict):
    temp_dataset = read_dataset("adult-hw1.csv")
    if high >= low:
        mid = (high + low) // 2
        print("mid", mid)
        if mid==low:
            anonym_dict = {}
            for node in lattice_dic[mid]:
                if k_anonym(node, temp_dataset, parents_dict, k):
                    anonym_dict[node] = lm_cost(node, raw_dataset, parents_dict, leaf_dict)
            if len(anonym_dict) != 0:
                return min(anonym_dict, key=anonym_dict.get) #choose the node yielding min lm cost
                
            else: #none of the nodes at level mid satisfies k-anonimity, then check one level up
                for node in lattice_dic[mid+1]:
                   # anonym_dataset = k_anonym(node, temp_dataset, parents_dict, k)
                    if k_anonym(node, temp_dataset, parents_dict, k):
                        anonym_dict[node] = lm_cost(node, raw_dataset, parents_dict, leaf_dict)
                print(mid)
                return min(anonym_dict, key=anonym_dict.get) #choose the node yielding min lm cost        
        for node in lattice_dic[mid]: #check the nodes at level mid
            if k_anonym(node, temp_dataset, parents_dict, k):
                return binary_search(lattice_dic, low, mid, temp_dataset, parents_dict, k, leaf_dict)
            
        return binary_search(lattice_dic, mid+1, high, temp_dataset, parents_dict, k, leaf_dict)
    return -1

def k_anonym(node, raw_dataset, parents_dict, k):
    temp_dataset = read_dataset("adult-hw1.csv")
    for t in node:
        temp_dataset = generalize_k_level(temp_dataset, t[1] , parents_dict, t[0])
    return check_group_sizes(pd.DataFrame(temp_dataset).groupby(list(temp_dataset[0].keys())[:-1]), k) 


def generalize_k_level(raw_dataset, k, parents_dict, field):
    if k==0:
        return raw_dataset
    temp_dataset = raw_dataset
    for i in range(len(temp_dataset)):
        parent_size = len(parents_dict[field][temp_dataset[i][field]]) #gives the parent list size
        if parent_size>k:
            general = parents_dict[field][temp_dataset[i][field]][parent_size-1-k]
            temp_dataset[i][field] = general
        else:
            general = parents_dict[field][temp_dataset[i][field]][0]
            temp_dataset[i][field] = general
    return temp_dataset

def check_group_sizes(df_group, k):
    check = True
    for name_of_group, contents_of_group in df_group:
        if len(contents_of_group) < k:
            check = False
    return check




def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    
    with open (DGH_file) as f:
        string = ""
        for line in f.readlines():
            counter = line.count("\t")
            word = " " * counter
            word += "'"
            for char in line:
                if char != "\t" and char != "\n":
                    word += char
            word += "':\n"
            string += word
        
    return yaml.safe_load(string)
    


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    level_dict = put_all_levels(DGHs)
    total_cost = 0
    for i in range(len(raw_dataset)): #for each record in datasets
        for kr,r,ka,a in zip([*raw_dataset[i]][:-1],[*raw_dataset[i].values()][:-1],[*anonymized_dataset[i]][:-1],[*anonymized_dataset[i].values()][:-1]): #for each key in record
            total_cost += abs(level_dict[kr][r]-level_dict[ka][a])
    return total_cost



def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    leaf_dict = leaf_calculator_all(DGHs)
    total_cost = 0
    qi_count = len(raw_dataset[0])-1
    for i in range(len(raw_dataset)): #for each record in datasets
        for kr,r,ka,a in zip([*raw_dataset[i]][:-1],[*raw_dataset[i].values()][:-1],[*anonymized_dataset[i]][:-1],[*anonymized_dataset[i].values()][:-1]): #for each key in record
            lm_cost = 0
            num1 = leaf_dict[ka][a]
            if num1==0:
                lm_cost = 0
            else:
                num2 = leaf_dict[ka]['Any']
                lm_cost = (num1-1)/(num2-1)
            total_cost += lm_cost/qi_count
    return total_cost         
            

def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    if D%k==0: #size of the dataset is a perfect multiple of k
        num = int(D/k)
        for i in range(num):
            mini_list = raw_dataset[i*k:i*k+k]
            clusters.append(mini_list)
    else:
        num = int(D//k-1)
        left = D-(D//k-1)*k
        for i in range(num):
            mini_list = raw_dataset[i*k:i*k+k]
            clusters.append(mini_list)
        clusters.append(raw_dataset[-left:])
        
    parents_dict = parent_creator_all(read_DGHs('./DGHs'))
    fields = clusters[0][0].keys()
    for cluster in clusters: #for each cluster do the same thing
        for field in fields:
            print(field)
            if field != 'income' and field!='index':             
                parent_lists = []
                for record in cluster:
                    parent_lists.append(parents_dict[field][record[field]])
                general_field = find_generalized(parent_lists)
                for record in cluster:
                    record[field] = general_field
    

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)



def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    
    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i
    anonymized_dataset = []
    marked = {} #dictionary holding the marked/unmarked info of the records, 0->unmarked 1->marked
    parents_dict = parent_creator_all(read_DGHs('./DGHs'))
    for i in range(len(raw_dataset)):
        marked[i] = 0
        
    clusters = []
    for i in range(len(raw_dataset)):
        if marked[i] == 0:
            marked[i] = 1
            current_r = raw_dataset[i]
            lm_costs = {}
            for j in range(1, len(raw_dataset)): #check the rest of the records and calculate the distances
                
                if marked[j] == 0:
                    #calculate lm cost for all attributes
                    next_r = raw_dataset[j]
                    total_cost = 0
                    for field in current_r.keys():
                        if field!='income' and field!='index':
                            l1 = parents_dict[field][current_r[field]]
                            l2 = parents_dict[field][next_r[field]]
                            general = find_generalized([l1,l2]) #finds the most specialized common parent of the records for that field
                            lm_cost = 0
                            num1 = leaf_dict[field][general] #number of descending leaves from that node
                            if num1==0:
                                lm_cost = 0
                            else:
                                num2 = leaf_dict[field]['Any'] #number of total leaves in that DGH
                                lm_cost = (num1-1)/(num2-1)/len(DGHs)
                            total_cost += lm_cost
                    lm_costs[j] = total_cost #keep the record of lm cost (distance from current record to the record at the index j)
               
            #now all distances to the following all unmarked records are calculated
            res = nsmallest(k-1, lm_costs, key = lm_costs.get)
            
            print(res)
            mini_list = []
            mini_list.append(current_r)
            for j in range(len(res)):   
              #  print(j)
                mini_list.append(raw_dataset[res[j]])
                marked[res[j]] = 1
            clusters.append(mini_list)
    
    fields = clusters[0][0].keys()
    for cluster in clusters: #for each cluster do the same thing
        for field in fields:
          #  print(field)
            if field != 'income' and field!='index':             
                parent_lists = []
                for record in cluster:
                    parent_lists.append(parents_dict[field][record[field]])
                general_field = find_generalized(parent_lists)
                for record in cluster:
                    record[field] = general_field
    
    # Finally, write dataset to a file
    #write_dataset(anonymized_dataset, output_file)
    anonymized_dataset = [None] * len(raw_dataset)
    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']
    write_dataset(anonymized_dataset, output_file)



def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    import time
    t_ = time.time()
    
    parents_dict = parent_creator_all(read_DGHs('./DGHs'))
    leaf_dict = leaf_calculator_all(DGHs)
    dic = {}
            
    level_dict = put_all_levels(read_DGHs('./DGHs'))
    levels_dict = {}

    for f in level_dict.keys():
        levels_dict[f] = max(level_dict[f].values())
        
    for key in levels_dict.keys():
        dic[key] = []
        n = levels_dict[key]
        for i in range(n+1):
            dic[key].append((key,i))
        
    lattice = []

    for el in itertools.product(*dic.values()):
        lattice.append(el)
        
    lattice_dic = {}

    for node in lattice:
        total = 0;
        for t in node:
            total += t[1]
        if total in lattice_dic:   
            lattice_dic[total].append(node)
        else:
            lattice_dic[total] = []
            lattice_dic[total].append(node)
            
    ##up until now, we have our generalization lattice
    total_levels = len(lattice_dic)
    chosen_node = binary_search(lattice_dic, 0, total_levels-1, raw_dataset, parents_dict, k, leaf_dict)  
          
    temp_dataset = read_dataset("adult-hw1.csv")
    for t in chosen_node:
        temp_dataset = generalize_k_level(temp_dataset, t[1] , parents_dict, t[0])

    # Finally, write dataset to a file
    write_dataset(temp_dataset, output_file)
    print("time", time.time()-t_)

    


# In[ ]:


import time
q = time.time()
bottomup_anonymizer("adult-hw1.csv", './DGHs', 128, 'bottomup-128.csv')
time.time() - q


# In[ ]:


# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now() ##
print(start_time) ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now() ##
print(end_time) ##
print(end_time - start_time)  ##

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300 5


# In[ ]:





# In[ ]:




