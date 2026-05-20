import joblib as jb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("air_quality.csv")


x = df[["City", "Air Quality", "Prominent Pollutant"]]
y = df["AQI Value"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

cat_cols = ["City", "Air Quality", "Prominent Pollutant"]

pre = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
)

model = Pipeline(
    [
        ("pre", pre),
        ("scaler", StandardScaler(with_mean=False)),
        ("lasso", Lasso(alpha=0.1)),
    ]
)

model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

mse = mean_squared_error(ytest, ypred)
rmse = root_mean_squared_error(ytest, ypred)

print("MSE:", mse)
print("RMSE:", rmse)

jb.dump(model, "lasso_model.pkl")
