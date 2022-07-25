# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #♥görselleştirmek için
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score    #başarı oranını verir.

data = pd.read_csv("insurance.csv")   #okuyacağımız dosya

print(data.columns)  #datayı okuma

# y ekseni
expenses = data.expenses.values.reshape(-1,1)

# x ekseni
ageBmis = data.iloc[:,[0,2]].values

regression = LinearRegression()
regression.fit(ageBmis,expenses)

print(regression.predict(np.array([[20,20],[30,21],[20,22],[20,23]])))
print(r2_score(expenses,regression.predict(ageBmis)))  #başarı oranı