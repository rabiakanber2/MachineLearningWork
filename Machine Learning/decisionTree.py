# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #♥görselleştirmek için
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score    #başarı oranını verir.

data = pd.read_csv("positions.csv")

level = data.iloc[:,1].values.reshape(-1,1)   #level okuma işlemi
salary = data.iloc[:,2].values.reshape(-1,1)  #salary okuma işlemi

#decisiontree

regression = DecisionTreeRegressor()
regression.fit(level,salary)
print(regression.predict([[8.3]]))
print(regression.predict([[8.9]]))

#grafik
plt.scatter(level,salary,color="red")
x = np.arange(min(level),max(level),0.01).reshape(-1,1)  #yapılandırma                
plt.plot(x,regression.predict(x),color="blue")
plt.xlabel("Level")    #x ekseni ismi
plt.ylabel("Salary")    # y ekseni ismi
plt.title("Decision Tree Model")  #grafik adı
plt.show() #grafiği göster

print(r2_score(salary,regression.predict(level)))  #başarı oranını yazdırma

#görselleştirme
