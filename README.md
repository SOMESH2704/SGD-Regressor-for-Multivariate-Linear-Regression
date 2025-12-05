# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Load California housing data, select features and targets, and split into training and testing sets. 2.Scale both X (features) and Y (targets) using StandardScaler. 3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data. 4.Predict on test data, inverse transform the results, and calculate the mean squared error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SOMESHWAR KUMAR
RegisterNumber:  212224240157

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


data = {
    'Size': [750, 800, 850, 900, 1200, 1500, 1700, 2000, 2200, 2500],
    'Rooms': [2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    'Location': [1, 2, 1, 3, 2, 3, 1, 2, 3, 2],
    'Price': [50, 55, 60, 70, 100, 130, 150, 200, 220, 250],
    'Occupants': [3, 3, 4, 4, 5, 6, 6, 7, 8, 9]
}

df = pd.DataFrame(data)


X = df[['Size', 'Rooms', 'Location']]
y = df[['Price', 'Occupants']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


price_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
occupants_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)


price_model.fit(X_train_scaled, y_train['Price'])
occupants_model.fit(X_train_scaled, y_train['Occupants'])


y_pred_price = price_model.predict(X_test_scaled)
y_pred_occupants = occupants_model.predict(X_test_scaled)


y_pred = np.column_stack((y_pred_price, y_pred_occupants))


print("Price Prediction - MSE:", mean_squared_error(y_test['Price'], y_pred_price))
print("Price Prediction - R2 Score:", r2_score(y_test['Price'], y_pred_price))

print("Occupants Prediction - MSE:", mean_squared_error(y_test['Occupants'], y_pred_occupants))
print("Occupants Prediction - R2 Score:", r2_score(y_test['Occupants'], y_pred_occupants))

print("\nActual values:\n", y_test.values)
print("\nPredicted values:\n", y_pred)

*/
```
## Output:


<img width="502" height="287" alt="image" src="https://github.com/user-attachments/assets/e17642c7-6336-4a52-a244-c1c91773a1f4" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
