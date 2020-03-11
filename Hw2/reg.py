# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 02:01:02 2020

@author: udarici19
"""

import numpy as np
import pandas as pd

df = pd.read_csv('data.csv').set_index('date')

#x = df[['gdppc','flfp']]
#y = df['fr']

def reg(x,y):
    x,y = dropnan(x,y)
    a = betas(x,y)
    b = standard_deviation(x,y)
    c = conf_interval(x,y)
    result = pd.concat([a,b,c], axis=1)
    result.rename(index={0: "Cons.", 1: "GDPPC", 2: "FLFP"}, inplace=True)
    return result

def dropnan(x,y):
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    conc = pd.concat([x,y], axis=1)
    conc = conc.dropna()
    x = conc.iloc[:,:-1]
    y = conc.iloc[:,-1:]
    return x,y

def xones(x):
    n = np.ones(len(x)).reshape(len(x),1)
    xones = np.concatenate([n,x], axis=1)
    return xones

def betas(x,y):
    #Beta Estimations Function after appending ones for the matrix calculations
    x = xones(x)
    beta_est = np.array(np.linalg.inv(x.transpose()@x)@x.transpose()@y) #b = (X'X)^-1X'y
    return pd.DataFrame(beta_est, columns = ["Cor_Coef"])

def standard_deviation(x,y):
    estimated_y = xones(x)@betas(x,y)
    errors = np.array(y) - estimated_y
    k = x.shape[1]
    sigma_square = (errors.transpose()@errors)/(len(x)- k -1)
    sigma_square = np.array(sigma_square)
    var = (np.linalg.inv(xones(x).transpose()@xones(x)))*sigma_square
    
    vars = []
    for i in range(len(var)):
        vars.append(var[i,i])
    std = np.sqrt(vars)
    
    return pd.DataFrame(std, columns=['Std_Dev'])
    
def conf_interval(x,y):
    m = betas(x,y)
    h = standard_deviation(x,y) * 1.96
    right_tail = m+np.array(h).reshape(3,1)
    left_tail = m-np.array(h).reshape(3,1)
    right_tail.rename(columns={"Cor_Coef": "Interval"}, inplace=True)
    left_tail.rename(columns={"Cor_Coef": "95% Conf. "}, inplace=True)
    return pd.concat([left_tail, right_tail], axis=1)


