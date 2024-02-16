import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


###############################################################################
############################### Label Flipping ################################
###############################################################################

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    total_acc_score = 0
    for i in range(100):
        count = round(n*len(y_train))
        indices_to_flip = random.sample(range(0, 100), count)   
        acc_score = 0
        y_train_copy = copy.deepcopy(y_train)
        for index in indices_to_flip:
            y_train_copy[index] = abs(y_train_copy[index]-1)
        model = None    
        if model_type == 'DT':
            model = DecisionTreeClassifier(max_depth=5, random_state=0)
        elif model_type == 'LR':
            model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
        elif model_type == 'SVC':
            model = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
        else:
            print('Wrong argument is given as model type.\n')
        
        model.fit(X_train, y_train_copy)
        predictions = model.predict(X_test)
        acc_score = accuracy_score(y_test, predictions)
        total_acc_score += acc_score 
        
    return total_acc_score/100

###############################################################################
############################## Inference ########################################
###############################################################################

def inference_attack(trained_model, samples, t):
    # TODO: You need to implement this function!  
    recall = 0
    true_positives = 0
    probs = trained_model.predict_proba(samples)
    for prob in probs: #check for each sample's probability distribution     
        if max(prob) >= t: #infer it as in training set
            true_positives += 1     
    recall = true_positives/len(samples)        
    return recall    

###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):    
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data

    #targeted label=1, target=1->0 with triggered pattern 43 as temperature
    X_train_v2 = copy.deepcopy(X_train)
    y_train_v2 = copy.deepcopy(y_train)
        
    label_1_indices = []
    for i in range(len(y_train_v2)):
        if y_train_v2[i]==1:
            label_1_indices.append(i)
            
    temp_indices = random.sample(range(0, len(label_1_indices)), num_samples)
    
    for index in temp_indices: #add trigger pattern into the sample in the chosen index
        chosen_index = label_1_indices[index]
        new_sample = X_train_v2[chosen_index]
        new_sample[0] = 43
        X_train_v2 = np.append(X_train_v2, np.array([new_sample]), axis=0)        
        y_train_v2 = np.append(y_train_v2, np.array([0]), axis=0)        
        
    if model_type == 'DT':
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == 'LR':
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    elif model_type == 'SVC':
        model = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    else:
        print('Wrong argument is given as model type.\n')
        return -1
        
    test_data = np.empty((0,len(X_train[0])), int)
    for i in range(200):
        temp = 43
        RH = random.randint(21,90)
        WS = random.randint(6,29)
        rain = random.uniform(0,16.8)
        FFMC = random.uniform(28.6,92.5)
        DMC = random.uniform(1.1,65.9)
        DC = random.uniform(7,220.4)
        ISI = random.uniform(0,18.5)
        BUI = random.uniform(1.1,68)
        FWU = random.uniform(0,31.1)
        
        sample = np.array([temp, RH, WS, rain, FFMC, DMC, DC, ISI, BUI, FWU])
        test_data = np.append(test_data, np.array([sample]), axis=0) 
    
    model.fit(X_train_v2, y_train_v2)
    backdoored_predict = model.predict(test_data)
    
    model.fit(X_train, y_train)
    normal_predict = model.predict(test_data)
        
    num_ones = 0
    num_successfull = 0
    for i in range(len(normal_predict)):
        if normal_predict[i] == 1:
            num_ones += 1
            if backdoored_predict[i] == 0:
                num_successfull += 1

    return num_successfull/num_ones



###############################################################################
############################## Evasion ########################################
###############################################################################

def evade_model(trained_model, actual_example):
    # TODO: You need to implement this function!
    actual_class = trained_model.predict([actual_example])[0]
    pred_class = actual_class
    modified_example = copy.deepcopy(actual_example)
    while pred_class == actual_class:
        # do something to modify the instance
        if actual_class == 1:  
            modified_example[4] -= 0.003
        else:
            modified_example[4] += 0.003
        pred_class = trained_model.predict([modified_example])[0]
        
    return modified_example

def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i]-adversarial_example[i])
        return tot/len(actual_example)

###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!
    print("Here, you need to conduct some experiments related to transferability and print their results...")
    
    modified_from_dt = np.empty((0,len(actual_examples[0])), int)
    modified_from_lr = np.empty((0,len(actual_examples[0])), int)
    modified_from_svc = np.empty((0,len(actual_examples[0])), int)

    for ex in actual_examples:
        modified_from_dt = np.append(modified_from_dt, np.array([evade_model(DTmodel, ex)]), axis=0)
        modified_from_lr = np.append(modified_from_lr, np.array([evade_model(LRmodel, ex)]), axis=0)
        modified_from_svc = np.append(modified_from_svc, np.array([evade_model(SVCmodel, ex)]), axis=0)
    
    DT_predicts_actual = DTmodel.predict(actual_examples)
    LR_predicts_actual = LRmodel.predict(actual_examples)
    SVC_predicts_actual = SVCmodel.predict(actual_examples)
    
    lr_from_dt_predicts = LRmodel.predict(modified_from_dt)
    svc_from_dt_predicts = SVCmodel.predict(modified_from_dt)
    
    dt_from_lr_predicts = DTmodel.predict(modified_from_lr)
    svc_from_lr_predicts = SVCmodel.predict(modified_from_lr)
    
    dt_from_svc_predicts = DTmodel.predict(modified_from_svc)
    lr_from_svc_predicts = LRmodel.predict(modified_from_svc)
    
    dt_to_lr = np.sum(np.absolute(np.subtract(lr_from_dt_predicts, LR_predicts_actual)))
    dt_to_svc = np.sum(np.absolute(np.subtract(svc_from_dt_predicts, SVC_predicts_actual)))
    
    lr_to_dt = np.sum(np.absolute(np.subtract(dt_from_lr_predicts, DT_predicts_actual)))
    lr_to_svc = np.sum(np.absolute(np.subtract(svc_from_lr_predicts, SVC_predicts_actual)))
    
    svc_to_dt = np.sum(np.absolute(np.subtract(dt_from_svc_predicts, DT_predicts_actual)))
    svc_to_lr = np.sum(np.absolute(np.subtract(lr_from_svc_predicts, LR_predicts_actual)))
    
    print("Out of 40 adversarial examples crafted to evade DT, " + str(dt_to_lr) + " of them transfer to LR.")
    print("Out of 40 adversarial examples crafted to evade DT, " + str(dt_to_svc) + " of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade LR, " + str(lr_to_dt) + " of them transfer to DT.")
    print("Out of 40 adversarial examples crafted to evade LR, " + str(lr_to_svc) + " of them transfer to SVC.")
    
    print("Out of 40 adversarial examples crafted to evade SVC, " + str(svc_to_dt) + " of them transfer to DT.")
    print("Out of 40 adversarial examples crafted to evade SVC, " + str(svc_to_lr) + " of them transfer to LR.")  

###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack

    model_predict = remote_model.predict(examples) #to be used as y_train, examples will be used as X_train
    if model_type == 'DT':
        stolen_model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == 'LR':
        stolen_model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    elif model_type == 'SVC':
        stolen_model = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    else:
        print('Wrong argument is given as model type.\n')
        return -1
        
    stolen_model.fit(examples, model_predict)
    return stolen_model
    

###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "forest_fires.csv"
    features = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
    
    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    df["DC"] = df["DC"].astype('float64')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)    
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 
    

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))
    
    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))
    
    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)
    
    # Inference attacks:
    samples = X_train[0:100]
    t_values = [0.99,0.98,0.96,0.8,0.7,0.5]
    for t in t_values:
        print("Recall of inference attack", str(t), ":", inference_attack(mySVC,samples,t))
    
    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)
    
    #Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"] 
    num_examples = 40
    for a,trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a] , ":" , total_perturb/num_examples)

    
    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])
    
    # Model stealing:
    budgets = [8, 12, 16, 20, 24]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))
    

if __name__ == "__main__":
    main()
