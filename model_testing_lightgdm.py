import joblib as jb
import pandas as pd

model = jb.load("lgbm_aqi_model.pkl")

data = pd.DataFrame(
    {
        "City": ["Delhi"],
        "Air Quality": ["Poor"],
        "Prominent Pollutant": ["PM2.5"],
    }
)

cat = ["City", "Air Quality", "Prominent Pollutant"]
for c in cat:
    data[c] = data[c].astype("category")

pred = model.predict(data)
print(pred)
