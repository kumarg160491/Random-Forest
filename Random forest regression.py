#import Liberaries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')
dataset

#make x and y for train and test dataset
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#linear regression 

from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(x_train,y_train)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10000,random_state=0)
regressor.fit(x,y)

#prediction
y_prediction=regressor.predict(x)


x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)

#visualising the training dataset
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.show()

