import numpy as np
import pandas as pd

df=pd.read_csv("/content/Concrete_Data_Yeh.csv")
df

df.isna().sum()

df.head()

df.tail()

df.shape

df.size

import matplotlib.pyplot as plt


# Features
features = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']

# Visualize each feature against the target variable 'csMPa'
for feature in features:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df['csMPa'], alpha=0.5)
    plt.title(f'{feature} vs. csMPa')
    plt.xlabel(feature)
    plt.ylabel('csMPa')
    plt.grid(True)
    plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = df.drop(columns=['csMPa'])  # Features
y = df['csMPa']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


from sklearn.metrics import mean_absolute_percentage_error
print("mape=",mean_absolute_percentage_error(y_test,y_pred))

from sklearn.metrics import r2_score
print('r2s=',r2_score(y_test,y_pred))

# Importing necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


# Generating sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)


# Initializing the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fitting the model to the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_pred = rf_regressor.predict(X_test)

# Calculating Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


from sklearn.metrics import r2_score
print('r2s=',r2_score(y_test,y_pred))
