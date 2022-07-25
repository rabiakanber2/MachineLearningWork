# -*- coding: utf-8 -*-

#doğrusal model burada doğru sonuç vermiyor.
#polinom doğrusal model kullanmalıyız.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #♥görselleştirmek için
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures   #polinom özelliklerini import eder. 

data = pd.read_csv("positions.csv")   #okuyacağımız dosya



print(data.columns)  #datayı okuma


# x ekseni
level = data.iloc[:,1].values.reshape(-1,1)

#y ekseni
salary = data.iloc[:,2].values.reshape(-1,1)

regression = LinearRegression()
regression.fit(level,salary)

tahmin = regression.predict([[8.3]])
print(tahmin)


regressionPoly = PolynomialFeatures(degree = 4)
levelPoly = regressionPoly.fit_transform(level) #level değerlerini polinom görğntğ haline getir.

regression2 = LinearRegression()
regression2.fit(levelPoly,salary)

tahmin2= regression2.predict(regressionPoly.fit_transform([[8.3]]))
print(tahmin2)


#görsellleştirme
plt.scatter(level,salary,color="red")

#grafik
plt.plot(level,regression.predict(level),color="blue")
plt.plot(level,regression.predict(levelPoly),color="black")
plt.show()        #!!!!!!!! HATA VAR X has 5 features, but LinearRegression is expecting 1 features as input.



