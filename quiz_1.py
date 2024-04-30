import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('energydata_complete.csv')

# From the dataset, fit a linear model on the relationship between the temperature
# in the living room in Celsius (x = T2) and the temperature outside the building (y = T6).
# What is the Root Mean Squared error in three D.P?
x = data[['T2']]
y = data['T6']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse.round(3))

# Remove the following columns: [“date”, “lights”]. The target variable is “Appliances”.
# Use a 70-30 train-test set split with a  random state of 42 (for reproducibility).
# Normalize the dataset using the MinMaxScaler (Hint: Use the MinMaxScaler fit_transform and
# transform methods on the train and test set respectively). Run a multiple linear regression
# using the training set. Answer the following questions:
x = data.drop(columns = ['date', 'lights', 'Appliances'])
y = data['Appliances']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

model = LinearRegression()
model.fit(scaled_x_train, y_train)
y_pred = model.predict(scaled_x_train)

# What is the Mean Absolute Error (in three decimal places) for the  training set?
mae_train = mean_absolute_error(y_train, y_pred)
print(mae_train.round(3))

# What is the Root Mean Squared Error (in three decimal places) for the training set?
mse_train = mean_squared_error(y_train, y_pred)
rmse_train = np.sqrt(mse_train)
print(rmse_train.round(3))

# What is the Mean Absolute Error (in three decimal places) for test set?
y_pred = model.predict(scaled_x_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(mae_test.round(3))

# What is the Root Mean Squared Error (in three decimal places) for test set?
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
print(rmse_test.round(3))

# Did the Model above overfit to the training set
diff_mae = mae_train - mae_test
print(diff_mae.round(3))
# Ans: No

# Train a ridge regression model with default parameters.
# Is there any change to the root mean squared error (RMSE) when evaluated on the test set?
ridge = Ridge()
ridge.fit(scaled_x_train, y_train)
y_pred = ridge.predict(scaled_x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse.round(3))
print(rmse - rmse_test)
# Ans: Yes

# Train a lasso regression model with default value and obtain the new feature weights with it.
# How many of the features have non-zero feature weights?
lasso_reg = Lasso()
lasso_reg.fit(scaled_x_train, y_train)
y_pred = lasso_reg.predict(scaled_x_test)

weights = pd.Series(data=lasso_reg.coef_, index=x_train.columns)
weights_df = pd.DataFrame(weights).reset_index()
weights_df.columns = ['Features', 'Lasso Regression']
weights_df['Lasso Regression'] = round(weights_df['Lasso Regression'], 3)
print(weights_df.groupby('Lasso Regression').count())

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse.round(3))