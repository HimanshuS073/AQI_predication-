import joblib as jb
import pandas as pd

model = jb.load("lasso_model.pkl")

data = pd.DataFrame(
    {
        "City": ["Delhi"],
        "Air Quality": ["Poor"],
        "Prominent Pollutant": ["PM2.5"],
    }
)

p = model.predict(data)

print(p)
