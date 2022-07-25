# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #♥görselleştirmek için
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("positions.csv")

level = data.iloc[:,1].values.reshape(-1,1)   #level okuma işlemi
salary = data.iloc[:,2].values  #salary okuma işlemi

regression = RandomForestRegressor(n_estimators=5,random_state=(0))
#n_estimators=5 kaç tane decisiontree oluşturayım belirlerken
#random_state=(0) ortaya çıkan sonucu değiştirme demek  
regression.fit(level,salary)

print(regression.predict([[8.3]]))
print(regression.predict([[8.9]]))