# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #♥görselleştirmek için
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score    #başarı oranını verir.


data = pd.read_csv("hw_25000.csv")   #okuyacağımız dosya

boy= data.Height.values.reshape(-1,1)
kilo= data.Weight.values.reshape(-1,1)

regression = LinearRegression()
regression.fit(boy,kilo)

print(regression.predict([[60]]))
print(regression.predict([[62]]))
print(regression.predict([[64]]))
print(regression.predict([[66]]))
print(regression.predict([[68]]))
print(regression.predict([[70]]))

print(data.columns)

plt.scatter(data.Height,data.Weight)

x = np.arange(min(data.Height),max(data.Height)).reshape(-1,1)  #grafik
plt.plot(x,regression.predict(x),color="red")  #grafik
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Simple Linear Regression Model")  #grafik başlığı

plt.show()

print(r2_score(kilo,regression.predict(boy)))  #başarı oranı
               
               
               
               
               
               