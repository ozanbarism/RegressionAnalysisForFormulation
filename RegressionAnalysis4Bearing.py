
"""

@author: Ozan Barış 61
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

#Data frame is inserted from excel
df=pd.read_excel("AlpTests.xlsx",sheet_name="All",usecols="F,H,I,K",header=28)

df.columns=("ShearStrain","BallVol","StressV","Qd")
df.drop(axis=0,index=108,inplace=True)
df.describe().T

X=df.iloc[:,0:3]
Y=df.iloc[:,3]

#a seaborn pairplot was conducted to see the relation of each variable with each other 
sns.pairplot(data=df)

#a training set and a testing set is defined with ratios %80 to %20 respectively.
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)

#Linear Regression is conducted
reg=LinearRegression().fit(X,Y)
Y_pred=[]
Y_pred=reg.predict(X)

#mae and mse are computed in by comparing y and y_predicted
print(mean_absolute_error(Y,Y_pred))
print(mean_squared_error(Y,Y_pred))

#Linear, ridge and lasso models are defined
linear_model=LinearRegression()
ridge_model=Ridge()
lasso_model=Lasso()

#weights are assigned with the training set
linear_model.fit(X_train,Y_train)
ridge_model.fit(X_train,Y_train)
lasso_model.fit(X_train,Y_train)

#R-square values are computed for each training set
print(linear_model.score(X_train,Y_train))
print(ridge_model.score(X_train,Y_train))
print(lasso_model.score(X_train,Y_train))

#predictions are done for testing data
lin_pred=linear_model.predict(X_test)
ridge_pred=ridge_model.predict(X_test)
lasso_pred=lasso_model.predict(X_test)

pred_dict={"Linear":lin_pred,"Ridge":ridge_pred,"Lasso":lasso_pred}

#statistical accuracy parameters are printed for the test dataset for linear,
#ridge and lasso models.
for key, value in pred_dict.items():
    print("Model:", key)
    print("R2 SCORE:", r2_score(Y_test,value))
    print("mean absolute error:", mean_absolute_error(Y_test,value))
    print("mean squared error:", mean_squared_error(Y_test,value))
    print("root mean squared error:", np.sqrt(mean_squared_error(Y_test,value)))
    print()