import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error

data=pd.read_csv("Ncov_data.csv")
data.iloc
X = data.iloc[:, 1:2].values
y = data.iloc[:, 0].values
x= data.iloc[:, 1].values
print(X)
print(y)
from sklearn.linear_model import LinearRegression

lin = LinearRegression()

lin.fit(X, y)

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
plt.scatter(X, y, color='blue')

plt.plot(X, lin.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('Cases')
plt.ylabel('Days')

plt.show()

plt.scatter(X, y, color='blue')


ypolyPred=lin2.predict(poly.fit_transform(X))

r2 = r2_score(y,ypolyPred)
rmse = np.sqrt(mean_squared_error(y,ypolyPred))



plt.plot(X, lin2.predict(poly.fit_transform(X)), color='red',label=r'Y=0.00000000e+00  1.79634449e-02X -7.85003235e-06$X^2  1.27140646e-09X^3$')
plt.title(' NCOV 2019 Polynomial Regression Analysis using 3rd Degree polynomial')
plt.xlabel('Cases')
plt.ylabel('Days')
plt.legend(loc='upper left')
plt.text(2000, 3, r'$R^2 = %0.2f$' % r2)
plt.text(2000, 2, 'RMSE= %0.2f' % rmse)
plt.show()
print(lin2.coef_)
print(r2)
print(rmse )
