import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("air_quality.csv")

x = df[["City", "Air Quality", "Prominent Pollutant"]]
y = df["AQI Value"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

cat = ["City", "Air Quality", "Prominent Pollutant"]

for c in cat:
    x_train[c] = x_train[c].astype("category")
    x_test[c] = x_test[c].astype("category")

m = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1,
)

m.fit(
    x_train,
    y_train,
    eval_set=[(x_test, y_test)],
    eval_metric="rmse",
    categorical_feature=cat,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

p = m.predict(x_test, num_iteration=m.best_iteration_)
r = np.sqrt(mean_squared_error(y_test, p))
print(r)

joblib.dump(m, "lgbm_aqi_model.pkl")
