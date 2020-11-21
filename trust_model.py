# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:39:31 2019

@author: amins
"""

import random
import csv
from sklearn.model_selection import train_test_split
import pyfpgrowth as fp
from itertools import combinations as com

#this function is to load the data from the file
def loadData(filename='dataset.txt'):
    try:
        with open(filename, 'r') as file:
            mylist = file.readlines()
            file.close()

        dataset = list()

        for x in mylist:
            x = x.replace('\n','').split(', ')
            dataset.append(x)

        return train_test_split(dataset, test_size=0.30);
    
    except FileNotFoundError:
        print(filename, "file does not exist, please provide a valid file")
        
#this function builds the rules of the TRUSTED and UNTRUSTED lists,
#from the frequent item
def build_rules(min_supp = 2):
    
    print('minimum support = ', min_supp)
    
    #get frequent item using fp-growth   
    tr = fp.find_frequent_patterns(tr_data, min_supp)
    utr = fp.find_frequent_patterns(untr_data, min_supp)
    
    #prepare the frequent item
    tr_rules = [set(x) for x in tr]
    utr_rules = [set(x) for x in utr]
    
    return tr_rules, utr_rules;

#this function prepares the dataset,
#by separate it into trustworth and untrustworthy list
def preData(dataset):
    
    tr_data = list ()
    utr_data = list ()

    for item in dataset:
        if 'untrustworthy' in item:
            temp1 = item.copy()
            temp1.remove('untrustworthy')
            utr_data.append(temp1)
        elif 'trustworthy' in item:
            temp2 = item.copy()
            temp2.remove('trustworthy')
            tr_data.append(temp2)
            
    return tr_data, utr_data;

#this function returns the total number of the entries,
#as well as the occurrence of them by the predefined rules
def dataStat(data_list, rules):
    occ = dict()
    for x in range(0, len(rules)):
        for y in range(0, len(data_list)):
            tt = rules[x].intersection(data_list[y])
            if(tt == rules[x] and tt != {}):
                e = str(tt)
                if(e not in occ):
                    occ[e] = 1
                else:
                    occ[e] += 1
    total = 0
    for each in occ:
        total += occ[each]
        
    return total, occ;

#this function classifies the data,
#by the Naive Bayes classifier.
#It retruens the class of the entry
def NaiveB(test, trusted_list, tr_total, tr_occ, untrusted_list, un_total, utr_occ):
    
    temp = list()
    for i in range(1, len(test)): # +1 in the len() for all, include the same list itself
        els = [set(x) for x in com(test, i)]
        temp.append(els)
        
    combs = list()
    for x in temp:
        [combs.append(each) for each in x]
        
    tn = len(trusted_list)
    un = len(untrusted_list)
    total = tn + un
    
    itemsets = list()
    [itemsets.append(x) for x in tr_occ if(x not in itemsets)]
    [itemsets.append(y) for y in utr_occ if(y not in itemsets)]

    D = len(itemsets)
    
    tnk = 0
    tts = 1
    for x in combs:
        if(str(x) in tr_occ):
            tnk = tr_occ[str(x)]
        else:
            tnk = 0
        tts *= (tnk+1)/(tr_total+D)

    tts *= (tn/total)

    unk = 0
    uts = 1
    for x in combs:
        if(str(x) in utr_occ):
            unk = utr_occ[str(x)]
        else:
            unk = 0
        uts *= (unk+1)/(un_total+D)

    uts *= (un/total)
    
    if(tts >= uts):
        return 'trustworthy'
        
    else:
        return 'untrustworthy'
#load the data from the file
training, testing = loadData()

#prepare the training data
tr_data, untr_data = preData(training)

min_supp = 1

#get the rules by providing a minimum support
tr_rules, utr_rules = build_rules(min_supp)

#get the total and the occurrance of the TRUSTED data
tr_total, tr_occ = dataStat(tr_data, tr_rules)

#get the total and the occurrance of the UNTRUSTED data
un_total, utr_occ = dataStat(untr_data, utr_rules)

count = 0
trans_max = 10

#open the file for recording the reuslt
with open('result.csv', 'a', newline='') as file:
    
    writer = csv.writer(file)
    writer.writerow(['min_supp_' + str(min_supp)])
    writer.writerow(['Number of transaction', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', \
                     'Area under ROC curve (AUC)'])
    
    while(count < 9): #loop through 
        
        TP = 0; FP = 0; TN = 0; FN = 0
        terminate = False
        trans = 0
        
        #prepare the testing data
        tr_test_data, utr_test_data = preData(testing)

        while(trans < trans_max):
            
            rn = random.randrange(3, 6)
            if rn > len(tr_test_data):
                rn = len(tr_test_data)
                terminate = True

            #test the data
            for j in range (0, rn):
                res = NaiveB(tr_test_data[j], tr_data, tr_total, tr_occ, untr_data, un_total, utr_occ)
                if res is 'trustworthy':
                    TP += 1
                else:
                    FN += 1
                trans += 1
                if trans >= trans_max:
                    break

            if trans >= trans_max:
                    break

            if rn > 0:
                del tr_test_data[0:rn]

            if len(utr_test_data) > 0:
                res = NaiveB(utr_test_data[0], tr_data, tr_total, tr_occ, untr_data, un_total, utr_occ)
                if res is 'untrustworthy':
                    TN += 1
                else:
                    FP += 1
                del utr_test_data[0]

                trans += 1
                if trans >= trans_max:
                    break

            if terminate == True:
                break

        #calculate the performance metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitiviy = (TP) / (TP + FN)
        specificity = (TN) / (TN + FP)
        precision = (TP) / (TP + FP)
        AUC = (sensitiviy + specificity) / 2

        print ('accuracy = ', accuracy, ', sensitiviy = ',  sensitiviy, ', specificity = ' , specificity, \
              ' precision = ', precision, ', AUC = ', AUC)
        writer.writerow([trans_max, accuracy, sensitiviy, specificity, precision, AUC])
        count += 1
        trans_max += 10
        
    writer.writerow(['\n'])
    file.close()
