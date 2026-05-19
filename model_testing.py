import joblib as jb
import pandas as pd

model = jb.load("model.jonlib")

data = pd.DataFrame(
    [{"year": 2026, "month": 5, "day": 19, "city": 34, "air": 2, "pollutant": 5}]
)

p = model.predict(data)

print(p)
