import joblib as jb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("air_quality.csv")

df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

df["city"] = le1.fit_transform(df["City"])
df["air"] = le2.fit_transform(df["Air Quality"])
df["pollutant"] = le3.fit_transform(df["Prominent Pollutant"])

X = df[["year", "month", "day", "city", "air", "pollutant"]]
y = df["AQI Value"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None
)

m = LinearRegression()
m.fit(x_train, y_train)

jb.dump(m, "model.jonlib")
p = m.predict(x_test)

mse = mean_squared_error(y_test, p)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
