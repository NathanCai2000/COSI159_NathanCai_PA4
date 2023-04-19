# -*- coding: utf-8 -*-
"""
@author: Nathan Cai
"""
import numpy as np
import pandas as pd

def adjust_csv(csv):
    csv = csv.replace(to_replace="Male", value="Man")
    csv = csv.replace(to_replace="Female", value="Woman")
    csv = csv.replace(to_replace=["East Asian", "Southeast Asian"], value="asian")
    csv = csv.replace(to_replace="Indian", value="indian")
    csv = csv.replace(to_replace="Black", value="black")
    csv = csv.replace(to_replace="White", value="white")
    csv = csv.replace(to_replace="Middle Eastern", value="middle eastern")
    csv = csv.replace(to_replace="Latino_Hispanic", value="latino hispanic")
    
    return csv
    
def comparison(a, b):
    if a == 'nan' or b == 'nan':
        return 2
    else:
        return int(a == b)
        
def sortit(data, true, sets):
    tmp = np.zeros([len(sets), 3])
    #[fail, pass, miss]
    for a in range(len(data)):
        b = comparison(data[a], true[a])        
        tmp[sets.index(true[a])][b] += 1
        
    retu = {'Class' : sets,
            'Fail' : tmp[:,0],
            'Pass' : tmp[:,1],
            'Miss' : tmp[:,2]
            }
    retu = pd.DataFrame(retu).set_index('Class')
    return retu
    
def ages(str):
    if str == 'more than 70':
        return [71, 200]
    else:
        a = str.find('-')
        b = str[:a]
        c = str[a+1:]
        if c.isnumeric():
            return [int(b), int(c)]
        else:
            return [int(b), int(b)]
    

def sortit_age(data, true, sets):
    tmp = np.zeros([len(sets), 3])
    #[fail, pass, miss]
    for a in range(len(data)):
        hld = ages(true[a])
        if data[a] == 'nan':
            b = 2
        else:
            b = int(data[a] >= hld[0] and data[a] <= hld[1])  
        tmp[sets.index(true[a])][b] += 1
        
    retu = {'Class' : sets,
            'Fail' : tmp[:,0],
            'Pass' : tmp[:,1],
            'Miss' : tmp[:,2]
            }
    retu = pd.DataFrame(retu).set_index('Class')
    return retu
    
def discriminate(data, true, s, sets):
    tmp = np.zeros([len(sets), 3])
    #[fail, pass, miss]
    for a in range(len(data)):
        hld = ages(true[a])
        if data[a] == 'nan':
            b = 2
        else:
            b = int(data[a] >= hld[0] and data[a] <= hld[1])     
        #print(g[a])
        tmp[sets.index(s[a])][b] += 1
        
    retu = {'Class' : sets,
            'Fail' : tmp[:,0],
            'Pass' : tmp[:,1],
            'Miss' : tmp[:,2]
            }
    retu = pd.DataFrame(retu).set_index('Class')
    return retu
    
def findacc(lst):
    retu = []
    for a in range(len(lst)):
        b = lst['Pass'][a]
        c = lst['Pass'][a] + lst['Fail'][a]# + lst['Miss'][a]
        retu.append(b/c)
    retu = np.round(np.array(retu),2) *100
    return retu.astype(int)
    
    