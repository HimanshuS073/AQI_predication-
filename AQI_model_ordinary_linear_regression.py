# imports important libraries to model testing on label encoder algorithm
import joblib as jb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# read data set in csv formate
df = pd.read_csv("air_quality.csv")

# this are the csv header to use
df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# encode categorical variable to store numeric values
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

# store coloums in each encoder variable
df["city"] = le1.fit_transform(df["City"])
df["air"] = le2.fit_transform(df["Air Quality"])
df["pollutant"] = le3.fit_transform(df["Prominent Pollutant"])

# split data into training and testing sets for 2d array
X = df[["year", "month", "day", "city", "air", "pollutant"]]
y = df["AQI Value"]

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None
)

# main linear regression model to train on the training data
m = LinearRegression()
m.fit(x_train, y_train)

# store the trained model using joblib
jb.dump(m, "ordinary_linear_regression_model.joblib")
p = m.predict(x_test)

# count mean squared error and root mean squared error
mse = mean_squared_error(y_test, p)
rmse = np.sqrt(mse)

# print the mean squared error and root mean squared error:
print("MSE:", mse)
print("RMSE:", rmse)

# comment and notes for this  model training code
# this model use a label encoder to convert categorical variables to numeric values
# BUT  lable encoder algorithm have issue of lable coparison and diffrence in label data
# so in next model we will use one hot encoder to avoid this issue
# also in next model we will use one hot encoder to convert categorical variables to numeric values
# also in next model we will use
